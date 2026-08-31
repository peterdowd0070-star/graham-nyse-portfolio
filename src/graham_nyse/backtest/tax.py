from __future__ import annotations

import copy
from dataclasses import dataclass

import pandas as pd

from graham_nyse.config import TaxConfig


@dataclass
class TaxLot:
    security_id: str
    acquired: pd.Timestamp
    quantity: float
    basis_per_share: float


@dataclass
class WashSaleWindow:
    security_id: str
    expires: pd.Timestamp
    remaining_quantity: float
    loss_per_share: float
    sale_year: int
    long_term: bool


TAX_MODES = {
    "tax_deferred": (False, "fifo", False),
    "taxable_fifo_no_liquidation": (True, "fifo", False),
    "taxable_hifo_no_liquidation": (True, "hifo", False),
    "taxable_hifo_terminal_liquidation": (True, "hifo", True),
}


class TaxLedger:
    """Deterministic lot ledger with annual netting and wash-sale basis adjustments."""

    def __init__(self, cfg: TaxConfig, mode: str):
        if mode not in TAX_MODES:
            raise ValueError(f"Unknown tax mode: {mode}")
        self.cfg = cfg
        self.mode = mode
        self.enabled, self.lot_method, self.terminal_liquidation = TAX_MODES[mode]
        self.lots: dict[str, list[TaxLot]] = {}
        self.wash_windows: list[WashSaleWindow] = []
        self.realized_short: dict[int, float] = {}
        self.realized_long: dict[int, float] = {}
        self.qualified_dividends: dict[int, float] = {}
        self.ordinary_dividends: dict[int, float] = {}
        self.short_loss_carryforward = 0.0
        self.long_loss_carryforward = 0.0
        self.settled_years: set[int] = set()
        self.tax_adjustments: dict[int, float] = {}
        self.tax_paid = 0.0
        self.events: list[dict[str, object]] = []

    def _prune_wash_windows(self, day: pd.Timestamp) -> None:
        self.wash_windows = [
            w
            for w in self.wash_windows
            if w.expires >= day and w.remaining_quantity > 1e-12
        ]

    def _restore_disallowed_loss(
        self, window: WashSaleWindow, amount: float, recognized_on: pd.Timestamp
    ) -> None:
        if window.sale_year in self.settled_years:
            rate = (
                self.cfg.rates.long_term
                if window.long_term
                else self.cfg.rates.short_term
            )
            self.tax_adjustments[recognized_on.year] = (
                self.tax_adjustments.get(recognized_on.year, 0.0) + amount * rate
            )
            return
        bucket = self.realized_long if window.long_term else self.realized_short
        bucket[window.sale_year] = bucket.get(window.sale_year, 0.0) + amount

    def buy(
        self, security_id: str, quantity: float, price: float, day: pd.Timestamp
    ) -> None:
        if quantity <= 0:
            return
        day = pd.Timestamp(day).normalize()
        adjusted_basis = float(price)
        if self.enabled:
            self._prune_wash_windows(day)
            remaining = quantity
            basis_adjustment = 0.0
            for window in self.wash_windows:
                if window.security_id != security_id or remaining <= 1e-12:
                    continue
                matched = min(remaining, window.remaining_quantity)
                disallowed = matched * window.loss_per_share
                basis_adjustment += disallowed
                self._restore_disallowed_loss(window, disallowed, day)
                remaining -= matched
                window.remaining_quantity -= matched
            adjusted_basis += basis_adjustment / quantity
        self.lots.setdefault(security_id, []).append(
            TaxLot(security_id, day, quantity, adjusted_basis)
        )
        self.events.append(
            {
                "date": day,
                "event": "BUY_LOT",
                "security_id": security_id,
                "quantity": quantity,
                "basis": adjusted_basis,
            }
        )

    def sell(
        self, security_id: str, quantity: float, price: float, day: pd.Timestamp
    ) -> float:
        if quantity <= 0:
            return 0.0
        day = pd.Timestamp(day).normalize()
        lots = self.lots.get(security_id, [])
        if sum(l.quantity for l in lots) + 1e-9 < quantity:
            raise ValueError(
                f"Tax lots do not support sale of {quantity} shares of {security_id}"
            )
        if self.lot_method == "hifo":
            lots.sort(key=lambda lot: (-lot.basis_per_share, lot.acquired))
        else:
            lots.sort(key=lambda lot: lot.acquired)
        remaining = quantity
        realized = 0.0
        for lot in lots:
            if remaining <= 1e-12:
                break
            sold = min(remaining, lot.quantity)
            gain = sold * (price - lot.basis_per_share)
            realized += gain
            holding_days = (day - lot.acquired).days
            long_term = holding_days > self.cfg.long_term_days
            bucket = self.realized_long if long_term else self.realized_short
            taxable_gain = gain
            if self.enabled and gain < 0:
                loss_per_share = -gain / sold
                unmatched = sold
                lower = day - pd.Timedelta(self.cfg.wash_sale_days, unit="D")
                for replacement in lots:
                    if (
                        replacement is lot
                        or replacement.quantity <= 1e-12
                        or not (lower <= replacement.acquired <= day)
                    ):
                        continue
                    matched = min(unmatched, replacement.quantity)
                    disallowed = matched * loss_per_share
                    replacement.basis_per_share += disallowed / replacement.quantity
                    taxable_gain += disallowed
                    unmatched -= matched
                    if unmatched <= 1e-12:
                        break
                if unmatched > 1e-12:
                    self.wash_windows.append(
                        WashSaleWindow(
                            security_id,
                            day + pd.Timedelta(self.cfg.wash_sale_days, unit="D"),
                            unmatched,
                            loss_per_share,
                            day.year,
                            long_term,
                        )
                    )
            bucket[day.year] = bucket.get(day.year, 0.0) + taxable_gain
            lot.quantity -= sold
            remaining -= sold
        self.lots[security_id] = [lot for lot in lots if lot.quantity > 1e-12]
        self.events.append(
            {
                "date": day,
                "event": "SELL_LOT",
                "security_id": security_id,
                "quantity": quantity,
                "price": price,
                "realized_gain": realized,
            }
        )
        return realized

    def dividend(
        self, security_id: str, amount: float, day: pd.Timestamp, qualified: bool
    ) -> None:
        if amount <= 0:
            return
        target = self.qualified_dividends if qualified else self.ordinary_dividends
        target[day.year] = target.get(day.year, 0.0) + amount
        self.events.append(
            {
                "date": day,
                "event": "DIVIDEND",
                "security_id": security_id,
                "amount": amount,
                "qualified": qualified,
            }
        )

    def settle_year(self, year: int) -> float:
        if not self.enabled:
            return 0.0
        short = self.realized_short.pop(year, 0.0) - self.short_loss_carryforward
        long = self.realized_long.pop(year, 0.0) - self.long_loss_carryforward
        self.short_loss_carryforward = self.long_loss_carryforward = 0.0
        if short > 0 > long:
            net = short + long
            short, long = (net, 0.0) if net >= 0 else (0.0, net)
        elif long > 0 > short:
            net = short + long
            short, long = (0.0, net) if net >= 0 else (net, 0.0)
        self.short_loss_carryforward = max(0.0, -short)
        self.long_loss_carryforward = max(0.0, -long)
        short_taxable = max(0.0, short)
        long_taxable = max(0.0, long)
        tax = (
            short_taxable * self.cfg.rates.short_term
            + long_taxable * self.cfg.rates.long_term
            + self.qualified_dividends.pop(year, 0.0)
            * self.cfg.rates.qualified_dividend
            + self.ordinary_dividends.pop(year, 0.0) * self.cfg.rates.ordinary_dividend
            + self.tax_adjustments.pop(year, 0.0)
        )
        tax = max(0.0, tax)
        self.tax_paid += tax
        self.settled_years.add(year)
        self.events.append(
            {
                "date": pd.Timestamp(f"{year}-12-31"),
                "event": "TAX_SETTLEMENT",
                "amount": tax,
            }
        )
        return tax

    def estimate_year_tax(self, year: int) -> float:
        """Return the current liability without mutating lots, carryforwards or events."""
        if not self.enabled:
            return 0.0
        return copy.deepcopy(self).settle_year(year)

    def lot_frame(self) -> pd.DataFrame:
        rows = [
            {
                "security_id": lot.security_id,
                "acquired": lot.acquired,
                "quantity": lot.quantity,
                "basis_per_share": lot.basis_per_share,
            }
            for lots in self.lots.values()
            for lot in lots
        ]
        return pd.DataFrame(
            rows, columns=["security_id", "acquired", "quantity", "basis_per_share"]
        )

    def event_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.events)

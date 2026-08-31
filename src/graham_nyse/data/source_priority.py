from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

SourceRole = Literal[
    "filing_vintages",
    "historical_membership",
    "current_symbol_reference",
    "prices_and_actions",
    "price_audit",
    "terminal_events",
]


@dataclass(frozen=True)
class SourcePriority:
    role: SourceRole
    priority: int
    source: str
    access: Literal["public", "free_key", "research_free", "commercial"]
    authoritative_for_role: bool
    empirical_limit: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


# Priority is role-specific. Authority for filings does not imply authority for
# security returns, identity continuity, or terminal values.
SOURCE_PRIORITIES: tuple[SourcePriority, ...] = (
    SourcePriority("filing_vintages", 10, "sec_edgar", "public", True),
    SourcePriority(
        "historical_membership",
        10,
        "alpha_vantage_listing_status",
        "free_key",
        False,
        "dated symbols are not permanent security identifiers",
    ),
    SourcePriority(
        "historical_membership", 90, "wrds_crsp", "commercial", True
    ),
    SourcePriority(
        "current_symbol_reference", 10, "nasdaq_trader", "public", True
    ),
    SourcePriority(
        "current_symbol_reference", 20, "sec_exchange_tickers", "public", True
    ),
    SourcePriority(
        "prices_and_actions",
        10,
        "yahoo_research",
        "research_free",
        False,
        "inactive-symbol coverage and usage rights are not certified",
    ),
    SourcePriority("prices_and_actions", 90, "wrds_crsp", "commercial", True),
    SourcePriority(
        "price_audit",
        10,
        "stooq",
        "research_free",
        False,
        "audit only; no permanent-ID or corporate-action contract",
    ),
    SourcePriority("terminal_events", 10, "sec_form_25_8k", "public", True),
    SourcePriority(
        "terminal_events",
        90,
        "wrds_crsp_delisting_returns",
        "commercial",
        True,
    ),
)


def source_plan(*, no_commercial: bool = True) -> dict[SourceRole, list[SourcePriority]]:
    plan: dict[SourceRole, list[SourcePriority]] = {}
    for item in SOURCE_PRIORITIES:
        if no_commercial and item.access == "commercial":
            continue
        plan.setdefault(item.role, []).append(item)
    for choices in plan.values():
        choices.sort(key=lambda item: item.priority)
    return plan


def source_plan_payload(*, no_commercial: bool = True) -> dict[str, object]:
    plan = source_plan(no_commercial=no_commercial)
    return {
        "mode": "no_commercial" if no_commercial else "all_sources",
        "roles": {
            role: [choice.as_dict() for choice in choices]
            for role, choices in sorted(plan.items())
        },
        "publication_rule": (
            "Source priority does not override dataset certification. "
            "Unresolved identity or terminal returns keep results research_only."
        ),
    }

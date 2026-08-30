from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

VINTAGE_KEYS = {"security_id", "accession_number", "accepted_at", "period_end"}


@dataclass(frozen=True)
class FilingVintageStore:
    """Immutable, acceptance-time-keyed normalized filing observations."""

    frame: pd.DataFrame

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> FilingVintageStore:
        missing = VINTAGE_KEYS - set(frame.columns)
        if missing:
            raise ValueError(f"Filing vintages are missing columns: {sorted(missing)}")
        out = frame.copy()
        out["security_id"] = out["security_id"].astype(str)
        out["accession_number"] = out["accession_number"].astype(str)
        out["accepted_at"] = pd.to_datetime(
            out["accepted_at"], utc=True, errors="raise"
        )
        out["period_end"] = pd.to_datetime(
            out["period_end"], errors="raise"
        ).dt.normalize()
        if out.duplicated(["security_id", "accession_number"]).any():
            raise ValueError("A filing accession may appear only once per security")
        return cls(
            out.sort_values(["accepted_at", "security_id"]).reset_index(drop=True)
        )

    def snapshot(self, decision_at: str | pd.Timestamp) -> pd.DataFrame:
        cutoff = pd.Timestamp(decision_at)
        cutoff = (
            cutoff.tz_localize("UTC")
            if cutoff.tzinfo is None
            else cutoff.tz_convert("UTC")
        )
        available = self.frame.loc[self.frame["accepted_at"].le(cutoff)].copy()
        if available.empty:
            return available
        # Later amendments become visible only after their own acceptance time.
        latest = (
            available.groupby("security_id", as_index=False, sort=False).tail(1).copy()
        )
        latest["snapshot_at"] = cutoff
        return latest.reset_index(drop=True)

    def new_since(
        self, previous_at: str | pd.Timestamp, decision_at: str | pd.Timestamp
    ) -> pd.DataFrame:
        previous = pd.Timestamp(previous_at)
        current = pd.Timestamp(decision_at)
        previous = (
            previous.tz_localize("UTC")
            if previous.tzinfo is None
            else previous.tz_convert("UTC")
        )
        current = (
            current.tz_localize("UTC")
            if current.tzinfo is None
            else current.tz_convert("UTC")
        )
        return self.frame.loc[
            self.frame["accepted_at"].gt(previous)
            & self.frame["accepted_at"].le(current)
        ].copy()

    def write_snapshot(
        self, decision_at: str | pd.Timestamp, output_dir: str | Path
    ) -> dict[str, object]:
        snapshot = self.snapshot(decision_at)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "filing_snapshot.parquet"
        snapshot.to_parquet(path, index=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {
            "decision_at": pd.Timestamp(decision_at).isoformat(),
            "row_count": len(snapshot),
            "sha256": digest,
            "maximum_accepted_at": None
            if snapshot.empty
            else snapshot["accepted_at"].max().isoformat(),
        }
        (out / "filing_snapshot_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest


def acceptance_metadata_from_submissions(payload: dict[str, Any]) -> pd.DataFrame:
    recent = payload.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame(
            columns=[
                "accession_number",
                "accepted_at",
                "filing_date",
                "report_date",
                "form",
            ]
        )
    frame = pd.DataFrame(recent)
    rename = {
        "accessionNumber": "accession_number",
        "acceptanceDateTime": "accepted_at",
        "filingDate": "filing_date",
        "reportDate": "report_date",
    }
    frame = frame.rename(columns=rename)
    required = ["accession_number", "accepted_at", "filing_date", "report_date", "form"]
    for column in required:
        if column not in frame:
            frame[column] = pd.NA
    frame["accepted_at"] = pd.to_datetime(
        frame["accepted_at"], utc=True, errors="coerce"
    )
    return frame[required].dropna(subset=["accession_number", "accepted_at"])

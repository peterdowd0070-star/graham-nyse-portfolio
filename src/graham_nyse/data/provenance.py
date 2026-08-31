from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_source_manifest(
    output_dir: str | Path,
    *,
    source: str,
    parameters: dict[str, Any],
    files: list[str | Path],
) -> Path:
    """Write non-secret provenance for a licensed or public-data extraction."""
    root = Path(output_dir)
    records = []
    for item in files:
        path = Path(item)
        records.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "source": source,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "parameters": parameters,
        "files": records,
    }
    target = root / "source_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return target

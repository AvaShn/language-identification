"""Safe artifact I/O and immutable-source integrity helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False, encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_manifest(root: Path, files: Iterable[Path]) -> dict[str, Any]:
    records = []
    for path in sorted({p.resolve() for p in files}):
        records.append(
            {
                "path": path.relative_to(root.resolve()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(records),
        "files": records,
    }


def compare_manifests(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    expected_map = {item["path"]: (item["bytes"], item["sha256"]) for item in expected["files"]}
    actual_map = {item["path"]: (item["bytes"], item["sha256"]) for item in actual["files"]}
    added = sorted(set(actual_map) - set(expected_map))
    removed = sorted(set(expected_map) - set(actual_map))
    changed = sorted(path for path in set(expected_map) & set(actual_map) if expected_map[path] != actual_map[path])
    return {"unchanged": not (added or removed or changed), "added": added, "removed": removed, "changed": changed}


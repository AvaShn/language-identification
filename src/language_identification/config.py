"""Configuration loading with repository-relative, machine-portable paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def find_project_root(start: Path | None = None) -> Path:
    """Find the closest parent containing ``pyproject.toml`` and ``configs``."""
    candidate = (start or Path.cwd()).resolve()
    for path in (candidate, *candidate.parents):
        if (path / "pyproject.toml").is_file() and (path / "configs").is_dir():
            return path
    raise FileNotFoundError(
        "Could not locate project root (expected pyproject.toml and configs/). "
        "Run the command from the repository or one of its subdirectories."
    )


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class ProjectConfig:
    """Resolved project configuration."""

    root: Path
    values: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.values["project"]["seed"])

    @property
    def test_size(self) -> float:
        return float(self.values["project"]["test_size"])

    @property
    def cv_folds(self) -> int:
        return int(self.values["project"]["cv_folds"])

    @property
    def n_jobs(self) -> int:
        return int(self.values["project"]["n_jobs"])

    @property
    def artifacts_dir(self) -> Path:
        return self.resolve_path(self.values["project"]["artifacts_dir"])

    @property
    def reports_dir(self) -> Path:
        return self.resolve_path(self.values["project"]["reports_dir"])

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.root / path).resolve()

    def section(self, name: str) -> dict[str, Any]:
        try:
            return dict(self.values[name])
        except KeyError as exc:
            raise KeyError(f"Missing required configuration section: {name}") from exc


def load_config(root: Path | None = None) -> ProjectConfig:
    """Load and merge the base, speech, and text YAML configuration files."""
    resolved_root = find_project_root(root)
    values: dict[str, Any] = {}
    for filename in ("base.yaml", "speech.yaml", "text.yaml"):
        path = resolved_root / "configs" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing required configuration file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Configuration must be a YAML mapping: {path}")
        values = _deep_merge(values, loaded)
    return ProjectConfig(root=resolved_root, values=values)


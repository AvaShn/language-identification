"""Environment and immutable-input preflight checks."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import psutil

from language_identification.common.io import (
    build_source_manifest,
    compare_manifests,
    ensure_dir,
    read_json,
    write_json,
)
from language_identification.config import ProjectConfig
from language_identification.speech.data import load_speech_data, speech_paths
from language_identification.text.data import load_text_data, text_source


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def run_preflight(config: ProjectConfig) -> dict[str, Any]:
    """Validate required local compute, packages, inputs, and output access."""
    speech = load_speech_data(config)
    text = load_text_data(config)
    speech_feature, speech_metadata = speech_paths(config)
    speech_files = [path for path in speech_feature.parent.rglob("*") if path.is_file()]
    text_files = [path for path in text_source(config).rglob("*") if path.is_file()]
    manifest = build_source_manifest(config.root, [*speech_files, *text_files])
    output = ensure_dir(config.artifacts_dir / "preflight")
    baseline_path = output / "input_manifest.json"
    if baseline_path.is_file():
        integrity = compare_manifests(read_json(baseline_path), manifest)
    else:
        write_json(baseline_path, manifest)
        integrity = {"unchanged": True, "added": [], "removed": [], "changed": [], "baseline_created": True}
    write_json(output / "source_integrity.json", integrity)
    if not integrity["unchanged"]:
        raise RuntimeError(f"Read-only input integrity check failed: {integrity}")
    probe = output / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    virtual_environment = Path(sys.prefix).resolve()
    report = {
        "status": "passed",
        "os": {"platform": platform.platform(), "system": platform.system(), "release": platform.release()},
        "python": {"version": sys.version, "executable": sys.executable, "prefix": sys.prefix},
        "poetry_environment": {
            "in_project_virtualenv": virtual_environment == (config.root / ".venv").resolve(),
            "virtual_env_variable": os.environ.get("VIRTUAL_ENV"),
            "lockfile_exists": (config.root / "poetry.lock").is_file(),
        },
        "compute": {
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "ram_gib": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "gpu_required": False,
            "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        },
        "packages": {
            name: _version(name)
            for name in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "PyYAML", "joblib", "nbclient", "nbformat", "ipykernel", "pytest")
        },
        "speech": {
            "feature_file": speech_feature.relative_to(config.root).as_posix(),
            "metadata_file": speech_metadata.relative_to(config.root).as_posix() if speech_metadata else None,
            "rows": len(speech.frame),
            "numeric_predictors": len(speech.feature_columns),
            "classes": sorted(set(speech.labels)),
            "read_only_boundary": True,
        },
        "text": {
            "source": text_source(config).relative_to(config.root).as_posix(),
            "rows_after_validation": len(text.frame),
            "classes": sorted(set(text.labels)),
            "independent_from_speech": True,
        },
        "outputs_writable": True,
        "source_integrity": integrity,
        "system_dependencies": {
            "required": [],
            "not_required": ["CUDA/GPU", "FFmpeg", "Conda", "paid/cloud APIs"],
        },
    }
    if not report["poetry_environment"]["in_project_virtualenv"]:
        report["status"] = "warning"
        report["action"] = "Run through Poetry after `poetry config virtualenvs.in-project true --local` and `poetry install`."
    write_json(output / "environment_report.json", report)
    return report


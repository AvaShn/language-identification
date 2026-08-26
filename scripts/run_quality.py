"""Run pytest inside the active Poetry environment and record machine-readable QA."""

from __future__ import annotations

import subprocess
import sys
from tempfile import TemporaryDirectory

from language_identification.common.io import ensure_dir, write_json
from language_identification.config import load_config


def run_tests() -> dict[str, object]:
    config = load_config()
    qa_dir = ensure_dir(config.artifacts_dir / "qa")
    with TemporaryDirectory(prefix="pytest-temp-", dir=qa_dir) as base_temp:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--basetemp", base_temp],
            cwd=config.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    result: dict[str, object] = {
        "status": "passed" if completed.returncode == 0 else "failed",
        "return_code": completed.returncode,
        "command": "python -m pytest (inside Poetry environment)",
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    write_json(qa_dir / "pytest.json", result)
    if completed.returncode != 0:
        raise RuntimeError(f"pytest failed\n{completed.stdout}\n{completed.stderr}")
    return result


if __name__ == "__main__":
    outcome = run_tests()
    print(outcome["stdout"])

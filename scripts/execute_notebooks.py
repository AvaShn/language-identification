"""Execute all required notebooks from clean kernels using the Poetry environment."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

from language_identification.common.io import write_json
from language_identification.common.report import NOTEBOOKS
from language_identification.config import load_config


def execute_required_notebooks() -> dict[str, str]:
    if sys.platform == "win32":
        # pyzmq requires add_reader support; the selector policy avoids a helper
        # thread and intermittent connection-reset noise during kernel shutdown.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    config = load_config()
    statuses: dict[str, str] = {}
    for relative in NOTEBOOKS:
        path = config.root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required notebook is missing: {path}")
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(config.root)}},
            allow_errors=False,
        )
        client.execute()
        nbformat.write(notebook, path)
        statuses[relative] = "executed"
    write_json(config.artifacts_dir / "qa" / "notebook_execution.json", statuses)
    return statuses


if __name__ == "__main__":
    result = execute_required_notebooks()
    print(f"Executed {len(result)} required notebooks")

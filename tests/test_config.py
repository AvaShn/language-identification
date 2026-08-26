from pathlib import Path

import pytest

from language_identification.config import find_project_root, load_config


def test_config_resolves_repository_relative_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root)
    assert config.root == root
    assert config.section("speech")["target_column"] == "lang"
    assert config.section("text")["raw_source"] != config.section("speech")["feature_source"]


def test_root_error_is_actionable() -> None:
    root = Path(__file__).resolve().parents[1]
    unrelated = Path(root.anchor) / "language-identification-missing-root-probe"
    with pytest.raises(FileNotFoundError, match="project root"):
        find_project_root(unrelated)

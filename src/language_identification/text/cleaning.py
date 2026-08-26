"""Conservative multilingual-safe text normalization."""

from __future__ import annotations

import re
from collections.abc import Iterable

from sklearn.base import BaseEstimator, TransformerMixin


def clean_text(value: str) -> str:
    """Collapse whitespace while preserving script, case, punctuation, and diacritics."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


class MultilingualTextCleaner(BaseEstimator, TransformerMixin):
    """Pickle-safe sklearn transformer applying :func:`clean_text`."""

    def fit(self, values: Iterable[str], y: object = None) -> "MultilingualTextCleaner":
        return self

    def transform(self, values: Iterable[str]) -> list[str]:
        return [clean_text(value) for value in values]

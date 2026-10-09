"""Gemeinsame Fixtures.

Tests für Funktionen, die noch nicht implementiert sind, tragen den Marker
``not_implemented``. Sie laufen als xfail, solange die Funktion
``NotImplementedError`` wirft, und werden zu normalen Tests, sobald die
Implementierung steht. Andere Fehler schlagen weiterhin fehl.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

not_implemented = pytest.mark.xfail(raises=NotImplementedError, reason="noch nicht implementiert")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20240101)


@pytest.fixture(scope="session")
def raa_triangle() -> pd.DataFrame:
    """RAA-Dreieck aus Mack (1993), kumuliert, Index Anfalljahr, Spalten 0..9."""
    cl = pytest.importorskip("chainladder")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = cl.load_sample("raa").to_frame(origin_as_datetime=False)
    frame.index = frame.index.year
    frame.columns = range(frame.shape[1])
    return frame

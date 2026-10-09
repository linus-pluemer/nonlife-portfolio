"""Tests für nonlife.reserving.

Gegenproben mit dem Paket chainladder auf dem RAA-Dreieck aus Mack (1993).
Beide Seiten rechnen deterministisch dieselbe Formel, daher enge Toleranz
(relativ 1e-9, nur Rundungsfehler der Gleitkommaarithmetik).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from nonlife.reserving import (
    calculate_development_factors,
    calculate_reserve,
    create_claims_triangle,
    mack_sigma2,
    mack_standard_error,
    project_triangle,
)
from tests.conftest import not_implemented

cl = pytest.importorskip("chainladder")


def _cl_fit() -> object:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cl.MackChainladder().fit(cl.load_sample("raa"))


# --- Aufbau des Dreiecks -----------------------------------------------------


def test_triangle_from_single_claims() -> None:
    claims = pd.DataFrame(
        {
            "accident_year": [2020, 2020, 2020, 2021, 2021, 2022],
            "development_year": [0, 1, 2, 0, 1, 0],
            "claim_amount": [100.0, 30.0, 10.0, 200.0, 50.0, 150.0],
        }
    )
    expected = pd.DataFrame(
        {0: [100.0, 200.0, 150.0], 1: [130.0, 250.0, np.nan], 2: [140.0, np.nan, np.nan]},
        index=pd.Index([2020, 2021, 2022], name="accident_year"),
    )
    triangle = create_claims_triangle(claims)
    pd.testing.assert_frame_equal(
        triangle, expected, check_names=False, check_column_type=False, check_dtype=False
    )


@pytest.mark.xfail(strict=True, reason="Befund Review: Zelle ohne Zahlung wird NaN statt Vorwert")
def test_triangle_cell_without_payment_carries_cumulative_value() -> None:
    # Anfalljahr 2020 hat in Abwicklungsjahr 1 keine Zahlung. Kumuliert muss dort
    # der Wert aus Abwicklungsjahr 0 stehen, nicht NaN.
    claims = pd.DataFrame(
        {
            "accident_year": [2020, 2020, 2021, 2021, 2022],
            "development_year": [0, 2, 0, 1, 0],
            "claim_amount": [100.0, 10.0, 200.0, 50.0, 150.0],
        }
    )
    triangle = create_claims_triangle(claims)
    assert triangle.loc[2020, 1] == pytest.approx(100.0)


# --- Chain-Ladder gegen chainladder -----------------------------------------


def test_development_factors_match_chainladder(raa_triangle: pd.DataFrame) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = cl.Development(average="volume").fit(cl.load_sample("raa")).ldf_.values  # noqa: PD011 (chainladder.Triangle, kein pandas)
    factors = calculate_development_factors(raa_triangle)
    np.testing.assert_allclose(factors, expected.ravel(), rtol=1e-9)


def test_reserve_matches_chainladder(raa_triangle: pd.DataFrame) -> None:
    factors = calculate_development_factors(raa_triangle)
    projected = project_triangle(raa_triangle, factors)
    reserve = calculate_reserve(raa_triangle, projected)

    summary = _cl_fit().summary_.to_frame(origin_as_datetime=False)  # type: ignore[attr-defined]
    expected_ibnr = summary["IBNR"].fillna(0.0).to_numpy()

    np.testing.assert_allclose(reserve["reserve_by_year"].to_numpy(), expected_ibnr, atol=1e-6)
    assert reserve["total_reserve"] == pytest.approx(expected_ibnr.sum(), rel=1e-9)
    assert reserve["projected_total"] == pytest.approx(summary["Ultimate"].sum(), rel=1e-9)


def test_projection_keeps_observed_cells(raa_triangle: pd.DataFrame) -> None:
    factors = calculate_development_factors(raa_triangle)
    projected = project_triangle(raa_triangle, factors)
    observed = raa_triangle.notna()
    pd.testing.assert_frame_equal(projected[observed], raa_triangle[observed])
    assert projected.notna().all().all()


def test_factor_one_gives_zero_reserve() -> None:
    # Ist jedes Anfalljahr ausentwickelt (Zeilen konstant), sind alle f_j = 1
    # und die Reserve ist 0.
    triangle = pd.DataFrame(
        {0: [10.0, 20.0, 30.0], 1: [10.0, 20.0, np.nan], 2: [10.0, np.nan, np.nan]},
        index=[2020, 2021, 2022],
    )
    factors = calculate_development_factors(triangle)
    np.testing.assert_allclose(factors, [1.0, 1.0])
    reserve = calculate_reserve(triangle, project_triangle(triangle, factors))
    assert reserve["total_reserve"] == pytest.approx(0.0)


# --- Mack ------------------------------------------------------------------


@not_implemented
def test_mack_sigma2_is_positive_and_complete(raa_triangle: pd.DataFrame) -> None:
    factors = calculate_development_factors(raa_triangle)
    sigma2 = mack_sigma2(raa_triangle, factors)
    assert sigma2.shape == factors.shape
    assert (sigma2 > 0).all()


@not_implemented
def test_mack_standard_error_matches_chainladder(raa_triangle: pd.DataFrame) -> None:
    factors = calculate_development_factors(raa_triangle)
    result = mack_standard_error(raa_triangle, factors)

    fit = _cl_fit()
    summary = fit.summary_.to_frame(origin_as_datetime=False)  # type: ignore[attr-defined]
    expected_se = summary["Mack Std Err"].fillna(0.0).to_numpy()
    by_year = result.drop(index="total")

    np.testing.assert_allclose(by_year["se"].to_numpy(dtype=float), expected_se, rtol=1e-6)
    np.testing.assert_allclose(
        by_year["reserve"].to_numpy(dtype=float), summary["IBNR"].fillna(0.0), rtol=1e-9
    )
    expected_total = float(fit.total_mack_std_err_.to_numpy().ravel()[0])  # type: ignore[attr-defined]
    assert result.loc["total", "se"] == pytest.approx(expected_total, rel=1e-6)


@not_implemented
def test_mack_total_se_exceeds_largest_single_se(raa_triangle: pd.DataFrame) -> None:
    # Die Reserven der Anfalljahre sind über gemeinsame f̂_j positiv korreliert,
    # daher ist se(Gesamt) ≥ max_i se(R_i).
    factors = calculate_development_factors(raa_triangle)
    result = mack_standard_error(raa_triangle, factors)
    assert result.loc["total", "se"] >= result.drop(index="total")["se"].max()

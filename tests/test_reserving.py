"""Tests für nonlife.reserving.

Gegenproben mit dem Paket chainladder auf dem RAA-Dreieck aus Mack (1993).
Beide Seiten rechnen deterministisch dieselbe Formel, daher enge Toleranz
(relativ 1e-9, nur Rundungsfehler der Gleitkommaarithmetik).
"""

from __future__ import annotations

import warnings
from dataclasses import replace

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
from nonlife.simulate import Portfolio, default_parameters, simulate_portfolio
from tests.conftest import not_implemented

cl = pytest.importorskip("chainladder")


def _cl_fit() -> object:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cl.MackChainladder().fit(cl.load_sample("raa"))


# --- Aufbau des Dreiecks -----------------------------------------------------


def _payments(accident_year: list[int], development_year: list[int],
              amount: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"accident_year": accident_year, "development_year": development_year, "amount": amount}
    )


@pytest.fixture(scope="module")
def portfolio() -> Portfolio:
    return simulate_portfolio(
        default_parameters(), 2_000, range(2015, 2025), np.random.default_rng(7)
    )


def test_triangle_from_payments() -> None:
    payments = _payments(
        [2020, 2020, 2020, 2021, 2021, 2022], [0, 1, 2, 0, 1, 0],
        [100.0, 30.0, 10.0, 200.0, 50.0, 150.0],
    )
    expected = pd.DataFrame(
        {0: [100.0, 200.0, 150.0], 1: [130.0, 250.0, np.nan], 2: [140.0, np.nan, np.nan]},
        index=pd.Index([2020, 2021, 2022], name="accident_year"),
    )
    triangle = create_claims_triangle(payments, valuation_year=2022)
    pd.testing.assert_frame_equal(
        triangle, expected, check_names=False, check_column_type=False, check_dtype=False
    )


def test_triangle_cell_without_payment_carries_cumulative_value() -> None:
    # Anfalljahr 2020 hat in Abwicklungsjahr 1 keine Zahlung. Kumuliert muss dort
    # der Wert aus Abwicklungsjahr 0 stehen, nicht NaN.
    payments = _payments(
        [2020, 2020, 2021, 2021, 2022], [0, 2, 0, 1, 0], [100.0, 10.0, 200.0, 50.0, 150.0]
    )
    triangle = create_claims_triangle(payments, valuation_year=2022)
    assert triangle.loc[2020, 1] == pytest.approx(100.0)
    assert triangle.loc[2020, 2] == pytest.approx(110.0)


def test_cells_after_valuation_are_nan() -> None:
    # Nur je eine Zahlung im ältesten Jahr: das NaN-Muster folgt allein aus i + j > v.
    triangle = create_claims_triangle(_payments([2020], [0], [1.0]), valuation_year=2023)
    i = triangle.index.to_numpy()[:, None]
    j = triangle.columns.to_numpy()
    np.testing.assert_array_equal(triangle.isna().to_numpy(), i + j > 2023)
    assert triangle.shape == (4, 4)


def test_accident_year_without_first_payment() -> None:
    # 2021 zahlt erst in Abwicklungsjahr 1: C_{2021,0} = 0, nicht NaN.
    payments = _payments([2020, 2021], [0, 1], [100.0, 40.0])
    triangle = create_claims_triangle(payments, valuation_year=2022)
    assert triangle.loc[2021, 0] == 0.0
    assert triangle.loc[2021, 1] == pytest.approx(40.0)
    assert triangle.loc[2022, 0] == 0.0


def test_rejects_payments_after_valuation() -> None:
    payments = _payments([2021, 2022], [2, 0], [10.0, 20.0])
    with pytest.raises(ValueError, match="Stichtag"):
        create_claims_triangle(payments, valuation_year=2022)


def test_latest_diagonal_equals_paid(portfolio: Portfolio) -> None:
    triangle = create_claims_triangle(portfolio.payments, portfolio.valuation_year)
    latest = triangle.ffill(axis=1).iloc[:, -1]
    paid = portfolio.payments.groupby("accident_year")["amount"].sum()
    np.testing.assert_allclose(latest.to_numpy(), paid.reindex(latest.index).to_numpy(),
                               rtol=1e-12)


def test_chain_ladder_exact_for_deterministic_pattern() -> None:
    # Mit Konzentration 1e8 zahlt jeder Schaden praktisch genau das Muster
    # (Streuung der Anteile ≈ √(p(1−p)/α₀) ≈ 1e-4). Dann ist jede Zeile
    # Endschaden · kumuliertes Muster, und Chain-Ladder trifft den wahren
    # Endschaden. Toleranz 1e-3 = Faktor 10 Puffer auf die Streuung eines
    # Einzelschadens; gemessen (Seed 7) sind es etwa 1e-5, da sich die
    # Abweichungen über die Schäden je Zelle mitteln.
    params = replace(default_parameters(), development_concentration=1e8)
    portfolio = simulate_portfolio(params, 2_000, range(2015, 2025), np.random.default_rng(7))
    triangle = create_claims_triangle(portfolio.payments, portfolio.valuation_year)
    factors = calculate_development_factors(triangle)
    reserve = calculate_reserve(triangle, project_triangle(triangle, factors))

    ultimate = project_triangle(triangle, factors).iloc[:, -1]
    np.testing.assert_allclose(ultimate.to_numpy(), portfolio.true_ultimate().to_numpy(),
                               rtol=1e-3)
    assert reserve["total_reserve"] == pytest.approx(portfolio.true_reserve().sum(), rel=1e-3)


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

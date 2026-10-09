"""Tests für nonlife.simulate.

Geprüft wird gegen die wahren Parameter der Simulation. Toleranzen sind
4 Standardfehler (Normalapproximation), bei festem Seed also deterministisch.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from nonlife.simulate import (
    Portfolio,
    TrueParameters,
    default_parameters,
    simulate_portfolio,
)

YEARS = range(2015, 2025)


@pytest.fixture(scope="module")
def portfolio() -> Portfolio:
    return simulate_portfolio(
        default_parameters(), n_policies=20_000, accident_years=YEARS,
        rng=np.random.default_rng(11),
    )


def test_parameters_are_consistent() -> None:
    params = default_parameters()
    assert isinstance(params, TrueParameters)
    assert sum(params.development_pattern) == pytest.approx(1.0)
    assert min(params.development_pattern) >= 0
    for factors in (*params.frequency_factors.values(), *params.severity_factors.values()):
        assert 1.0 in factors.values(), "jedes Merkmal braucht eine Referenzklasse mit Faktor 1"
    assert params.frequency_factors.keys() == params.severity_factors.keys()


def test_reproducible_with_same_generator_seed() -> None:
    def run(seed: int) -> Portfolio:
        return simulate_portfolio(
            default_parameters(), 500, YEARS, np.random.default_rng(seed)
        )

    a, b = run(5), run(5)
    pd.testing.assert_frame_equal(a.claims, b.claims)
    pd.testing.assert_frame_equal(a.payments, b.payments)
    assert not a.claims.equals(run(6).claims)


def test_payments_respect_valuation_date(portfolio: Portfolio) -> None:
    p = portfolio.payments
    assert portfolio.valuation_year == max(YEARS)
    assert (p["development_year"] >= 0).all()
    assert (p["accident_year"] + p["development_year"] <= portfolio.valuation_year).all()


def test_payments_never_exceed_claim(portfolio: Portfolio) -> None:
    paid = portfolio.payments.groupby("claim_id")["amount"].sum()
    amount = portfolio.claims.set_index("claim_id")["claim_amount"]
    assert (paid <= amount.loc[paid.index] * (1 + 1e-12)).all()


def test_oldest_year_almost_fully_paid(portfolio: Portfolio) -> None:
    # Mit 10 Abwicklungsjahren ist das älteste Anfalljahr bis auf den Rest des
    # Musters jenseits von Jahr 9 bezahlt.
    params = portfolio.params
    outstanding_share = sum(params.development_pattern[len(YEARS):])
    oldest = min(YEARS)
    paid = portfolio.payments.loc[portfolio.payments["accident_year"] == oldest, "amount"].sum()
    ultimate = portfolio.true_ultimate().loc[oldest]
    assert paid / ultimate == pytest.approx(1 - outstanding_share, abs=0.02)


def test_true_ultimate_is_sum_of_claims(portfolio: Portfolio) -> None:
    expected = portfolio.claims.groupby("accident_year")["claim_amount"].sum()
    pd.testing.assert_series_equal(portfolio.true_ultimate(), expected, check_names=False)


def test_frequency_per_level_matches_true_factor(portfolio: Portfolio) -> None:
    # Je Ausprägung ℓ von Merkmal m ist N_ℓ Poisson-verteilt mit Erwartung
    # Σ_i λ(x_i)·e_i. Prüfung: |N_ℓ - E| < 4·√E (Normalapproximation).
    params = portfolio.params
    pol = portfolio.policies
    lam = params.base_frequency * pol["exposure"].to_numpy()
    for feature, factors in params.frequency_factors.items():
        lam = lam * pol[feature].map(factors).to_numpy()
    expected_total = lam.sum()

    counts = portfolio.claims.groupby("policy_id").size().reindex(pol["policy_id"], fill_value=0)
    assert abs(counts.sum() - expected_total) < 4 * np.sqrt(expected_total)

    for feature in params.frequency_factors:
        by_level = pd.DataFrame(
            {"level": pol[feature].to_numpy(), "lam": lam, "n": counts.to_numpy()}
        )
        agg = by_level.groupby("level")[["lam", "n"]].sum()
        assert (np.abs(agg["n"] - agg["lam"]) < 4 * np.sqrt(agg["lam"])).all(), feature


def test_severity_mean_matches_true_parameters(portfolio: Portfolio) -> None:
    # Mittel von n Schäden: Standardfehler ≈ CV·E[X]/√n.
    params = portfolio.params
    claims = portfolio.claims.merge(portfolio.policies, on=["policy_id", "accident_year"])
    mean = np.full(len(claims), params.base_severity)
    for feature, factors in params.severity_factors.items():
        mean = mean * claims[feature].map(factors).to_numpy()
    ratio = claims["claim_amount"].to_numpy() / mean
    se = params.severity_cv / np.sqrt(len(claims))
    assert abs(ratio.mean() - 1.0) < 4 * se


@pytest.mark.parametrize(
    "change",
    [
        {"development_pattern": (0.5, 0.4)},
        {"development_pattern": (1.1, -0.1)},
        {"severity_cv": 0.0},
        {"full_year_share": 1.5},
        {"severity_factors": {**default_parameters().severity_factors, "region": {"urban": 1.2}}},
    ],
)
def test_invalid_parameters_are_rejected(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(default_parameters(), **change)


def test_exposure_in_unit_interval(portfolio: Portfolio) -> None:
    exposure = portfolio.policies["exposure"]
    assert ((exposure > 0) & (exposure <= 1)).all()
    share_full = (exposure == 1).mean()
    se = np.sqrt(0.7 * 0.3 / len(exposure))
    assert abs(share_full - portfolio.params.full_year_share) < 4 * se


def test_conditional_feature_distribution(portfolio: Portfolio) -> None:
    # Anteil "high" je Altersklasse gegen die bedingte Wahrscheinlichkeit.
    conditional = portfolio.params.conditional_features["vehicle_power"]
    pol = portfolio.policies
    for age, probabilities in conditional.probabilities.items():
        is_high = pol.loc[pol["driver_age"] == age, "vehicle_power"] == "high"
        p = probabilities["high"]
        assert abs(is_high.mean() - p) < 4 * np.sqrt(p * (1 - p) / len(is_high)), age


def test_fully_developed_claims_are_fully_paid(portfolio: Portfolio) -> None:
    last_year = portfolio.valuation_year - len(portfolio.params.development_pattern) + 1
    claims = portfolio.claims[portfolio.claims["accident_year"] <= last_year]
    paid = portfolio.payments.groupby("claim_id")["amount"].sum().loc[claims["claim_id"]]
    np.testing.assert_allclose(paid.to_numpy(), claims["claim_amount"].to_numpy(), rtol=1e-12)


def test_payment_shares_follow_pattern(portfolio: Portfolio) -> None:
    # Dirichlet(κp): E[Anteil_j] = p_j, Var[Anteil_j] = p_j(1 - p_j)/(κ + 1).
    params = portfolio.params
    oldest = portfolio.claims[portfolio.claims["accident_year"] == min(YEARS)]
    payments = portfolio.payments.merge(oldest[["claim_id", "claim_amount"]], on="claim_id")
    share = (payments["amount"] / payments["claim_amount"]).groupby(payments["development_year"])
    pattern = np.asarray(params.development_pattern)
    se = np.sqrt(pattern * (1 - pattern) / (params.development_concentration + 1) / len(oldest))
    assert (np.abs(share.mean().to_numpy() - pattern) < 4 * se).all()


def test_true_reserve_is_ultimate_minus_paid(portfolio: Portfolio) -> None:
    reserve = portfolio.true_reserve()
    paid = portfolio.payments.groupby("accident_year")["amount"].sum()
    np.testing.assert_allclose((reserve + paid).to_numpy(), portfolio.true_ultimate().to_numpy())
    assert reserve.iloc[0] == pytest.approx(0.0, abs=1e-6 * portfolio.true_ultimate().iloc[0])
    assert reserve.is_monotonic_increasing

"""Tests für nonlife.simulate.

Der erste Block prüft die bisherige Simulation. Der zweite Block beschreibt
die neue Schnittstelle (``simulate_portfolio``) und läuft als xfail, bis sie
implementiert ist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nonlife.simulate import (
    Portfolio,
    TrueParameters,
    default_parameters,
    simulate_claims,
    simulate_policies,
    simulate_portfolio,
)
from tests.conftest import not_implemented

# --- bisherige Simulation ----------------------------------------------------


def test_simulate_policies_shape() -> None:
    policies = simulate_policies(n_policies=200, seed=1)
    assert len(policies) == 200
    assert set(policies.columns) == {
        "policy_id",
        "exposure",
        "fahrer_alter",
        "ps_klasse",
        "region",
        "erwartete_frequenz",
    }


def test_exposure_range() -> None:
    policies = simulate_policies(n_policies=500, seed=1)
    assert policies["exposure"].between(0, 1).all()


def test_erwartete_frequenz_positiv() -> None:
    policies = simulate_policies(n_policies=500, seed=1)
    assert (policies["erwartete_frequenz"] > 0).all()


def test_claims_referenzieren_gueltige_policen() -> None:
    policies = simulate_policies(n_policies=500, seed=1)
    claims = simulate_claims(policies, seed=1)
    assert claims["policy_id"].isin(policies["policy_id"]).all()


def test_claim_amounts_positiv() -> None:
    policies = simulate_policies(n_policies=500, seed=1)
    claims = simulate_claims(policies, seed=1)
    assert (claims["claim_amount"] > 0).all()


def test_reproduzierbarkeit_mit_seed() -> None:
    p1 = simulate_policies(n_policies=100, seed=7)
    p2 = simulate_policies(n_policies=100, seed=7)
    assert p1.equals(p2)


# --- neue Schnittstelle ------------------------------------------------------

YEARS = range(2015, 2025)


@pytest.fixture(scope="module")
def portfolio() -> Portfolio:
    return simulate_portfolio(
        default_parameters(), n_policies=20_000, accident_years=YEARS,
        rng=np.random.default_rng(11),
    )


@not_implemented
def test_parameters_are_consistent() -> None:
    params = default_parameters()
    assert isinstance(params, TrueParameters)
    assert sum(params.development_pattern) == pytest.approx(1.0)
    assert min(params.development_pattern) >= 0
    for factors in (*params.frequency_factors.values(), *params.severity_factors.values()):
        assert 1.0 in factors.values(), "jedes Merkmal braucht eine Referenzklasse mit Faktor 1"
    assert params.frequency_factors.keys() == params.severity_factors.keys()


@not_implemented
def test_reproducible_with_same_generator_seed() -> None:
    def run(seed: int) -> Portfolio:
        return simulate_portfolio(
            default_parameters(), 500, YEARS, np.random.default_rng(seed)
        )

    a, b = run(5), run(5)
    pd.testing.assert_frame_equal(a.claims, b.claims)
    pd.testing.assert_frame_equal(a.payments, b.payments)
    assert not a.claims.equals(run(6).claims)


@not_implemented
def test_payments_respect_valuation_date(portfolio: Portfolio) -> None:
    p = portfolio.payments
    assert portfolio.valuation_year == max(YEARS)
    assert (p["development_year"] >= 0).all()
    assert (p["accident_year"] + p["development_year"] <= portfolio.valuation_year).all()


@not_implemented
def test_payments_never_exceed_claim(portfolio: Portfolio) -> None:
    paid = portfolio.payments.groupby("claim_id")["amount"].sum()
    amount = portfolio.claims.set_index("claim_id")["claim_amount"]
    assert (paid <= amount.loc[paid.index] * (1 + 1e-12)).all()


@not_implemented
def test_oldest_year_almost_fully_paid(portfolio: Portfolio) -> None:
    # Mit 10 Abwicklungsjahren ist das älteste Anfalljahr bis auf den Rest des
    # Musters jenseits von Jahr 9 bezahlt.
    params = portfolio.params
    outstanding_share = sum(params.development_pattern[len(YEARS):])
    oldest = min(YEARS)
    paid = portfolio.payments.loc[portfolio.payments["accident_year"] == oldest, "amount"].sum()
    ultimate = portfolio.true_ultimate().loc[oldest]
    assert paid / ultimate == pytest.approx(1 - outstanding_share, abs=0.02)


@not_implemented
def test_true_ultimate_is_sum_of_claims(portfolio: Portfolio) -> None:
    expected = portfolio.claims.groupby("accident_year")["claim_amount"].sum()
    pd.testing.assert_series_equal(portfolio.true_ultimate(), expected, check_names=False)


@not_implemented
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


@not_implemented
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

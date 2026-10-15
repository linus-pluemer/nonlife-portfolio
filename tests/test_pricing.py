"""Tests für nonlife.pricing.

Drei Arten von Prüfungen:

* analytisch: Bei einem einzigen kategorialen Merkmal ist das GLM saturiert
  in den Klassen, und der ML-Schätzer hat eine geschlossene Form. Für Poisson
  mit Offset ist exp(β̂_ℓ) = (N_ℓ/E_ℓ) / (N_ref/E_ref), für Gamma mit Log-Link
  ist exp(γ̂_ℓ) = X̄_ℓ / X̄_ref. Toleranz 1e-6 relativ (Abbruchkriterium von IRLS).
* gegen wahre Faktoren: |β̂ - β| < 4·se je Koeffizient. Bei 14 Koeffizienten
  wäre ein 99-%-Intervall je Koeffizient gemeinsam nur zu etwa 87 % erfüllt;
  4 se entsprechen gemeinsam mehr als 99,9 %. Mit festem Seed deterministisch.
* Format und Datenaufbereitung: exakt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nonlife.pricing import (
    fit_frequency_glm,
    fit_severity_glm,
    frequency_table,
    one_way_relativities,
    relativities,
    severity_table,
)
from nonlife.simulate import Portfolio, default_parameters, simulate_portfolio

TRUE_FREQ = {"Land": 1.0, "Stadt": 1.4, "Vorort": 1.1}
TRUE_SEV = {"Land": 1.0, "Stadt": 0.9, "Vorort": 1.2}


def _frequency_data(rng: np.random.Generator, n: int) -> pd.DataFrame:
    region = rng.choice(list(TRUE_FREQ), size=n)
    exposure = rng.uniform(0.25, 1.0, size=n)
    lam = 0.1 * np.vectorize(TRUE_FREQ.get)(region) * exposure
    return pd.DataFrame({"region": region, "exposure": exposure, "n_claims": rng.poisson(lam)})


def _severity_data(rng: np.random.Generator, n: int) -> pd.DataFrame:
    region = rng.choice(list(TRUE_SEV), size=n)
    mean = 2000.0 * np.vectorize(TRUE_SEV.get)(region)
    shape = 2.0
    return pd.DataFrame({"region": region, "claim_amount": rng.gamma(shape, mean / shape)})


def _within_4_se(model_rel: pd.DataFrame, true: dict[str, float], alpha: float) -> bool:
    # Aus dem (1 - alpha)-Intervall auf der Log-Skala zurückgerechnet: se = (log u - log l) / 2z.
    z = stats.norm.ppf(1 - alpha / 2)
    log_f = np.log(model_rel["factor"])
    se = (np.log(model_rel["upper"]) - np.log(model_rel["lower"])) / (2 * z)
    truth = np.log([true[level] for level in model_rel.index.get_level_values("level")])
    return bool((np.abs(log_f - truth) <= 4 * se).all())


# --- analytisch --------------------------------------------------------------


def test_poisson_relativities_equal_frequency_ratios(rng: np.random.Generator) -> None:
    data = _frequency_data(rng, 5_000)
    rel = relativities(fit_frequency_glm(data, ["region"]))

    by_region = data.groupby("region")[["n_claims", "exposure"]].sum()
    freq = by_region["n_claims"] / by_region["exposure"]
    for level in ("Stadt", "Vorort"):
        expected = freq[level] / freq["Land"]
        assert rel.loc[("region", level), "factor"] == pytest.approx(expected, rel=1e-6)


def test_gamma_relativities_equal_mean_ratios(rng: np.random.Generator) -> None:
    data = _severity_data(rng, 3_000)
    rel = relativities(fit_severity_glm(data, ["region"]))

    means = data.groupby("region")["claim_amount"].mean()
    for level in ("Stadt", "Vorort"):
        assert rel.loc[("region", level), "factor"] == pytest.approx(
            means[level] / means["Land"], rel=1e-6
        )


def test_reference_can_be_chosen(rng: np.random.Generator) -> None:
    data = _frequency_data(rng, 5_000)
    default = relativities(fit_frequency_glm(data, ["region"]))
    city = relativities(fit_frequency_glm(data, ["region"], reference={"region": "Stadt"}))

    assert city.loc[("region", "Stadt"), "factor"] == 1.0
    assert city.loc[("region", "Land"), "factor"] == pytest.approx(
        1 / default.loc[("region", "Stadt"), "factor"], rel=1e-6
    )


def test_unknown_reference_is_rejected(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError, match="Referenz"):
        fit_frequency_glm(_frequency_data(rng, 500), ["region"], reference={"region": "Mond"})


def test_one_way_equals_single_feature_glm(rng: np.random.Generator) -> None:
    data = _frequency_data(rng, 5_000).assign(policy_id=lambda d: np.arange(len(d)))
    claims = pd.DataFrame({"policy_id": np.repeat(data["policy_id"], data["n_claims"])})
    one_way = one_way_relativities(data, claims, "region", reference="Land")
    glm = relativities(fit_frequency_glm(data, ["region"]))
    for level in TRUE_FREQ:
        assert one_way[level] == pytest.approx(glm.loc[("region", level), "factor"], rel=1e-6)


# --- Format von relativities -------------------------------------------------


def test_reference_level_has_factor_one(rng: np.random.Generator) -> None:
    rel = relativities(fit_frequency_glm(_frequency_data(rng, 2_000), ["region"]))
    assert tuple(rel.loc[("region", "Land")]) == (1.0, 1.0, 1.0)
    assert list(rel.columns) == ["factor", "lower", "upper"]


def test_confidence_interval_contains_estimate(rng: np.random.Generator) -> None:
    rel = relativities(fit_frequency_glm(_frequency_data(rng, 2_000), ["region"]), alpha=0.05)
    assert (rel["lower"] <= rel["factor"]).all()
    assert (rel["factor"] <= rel["upper"]).all()


def test_wider_interval_for_smaller_alpha(rng: np.random.Generator) -> None:
    result = fit_frequency_glm(_frequency_data(rng, 2_000), ["region"])
    narrow, wide = relativities(result, alpha=0.10), relativities(result, alpha=0.01)
    assert (wide["upper"] >= narrow["upper"]).all()
    assert (wide["lower"] <= narrow["lower"]).all()


# --- Datenaufbereitung -------------------------------------------------------


@pytest.fixture
def small_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    policies = pd.DataFrame(
        {
            "policy_id": [1, 2, 3],
            "accident_year": [2024, 2024, 2024],
            "exposure": [1.0, 0.5, 1.0],
            "region": ["Land", "Stadt", "Stadt"],
        }
    )
    claims = pd.DataFrame(
        {
            "claim_id": [10, 11, 12],
            "policy_id": [1, 1, 3],
            "accident_year": [2024, 2024, 2024],
            "claim_amount": [100.0, 200.0, 300.0],
        }
    )
    return policies, claims


def test_frequency_table_counts_zero_claims(
    small_tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    policies, claims = small_tables
    table = frequency_table(policies, claims)
    assert table["n_claims"].tolist() == [2, 0, 1]
    assert len(table) == len(policies)


def test_severity_table_one_row_per_claim(
    small_tables: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    policies, claims = small_tables
    table = severity_table(policies, claims)
    assert len(table) == len(claims)
    assert table["region"].tolist() == ["Land", "Land", "Stadt"]
    assert not table.isna().any().any()


# --- gegen wahre Faktoren ----------------------------------------------------


def test_true_frequency_factors_within_4_se(rng: np.random.Generator) -> None:
    rel = relativities(fit_frequency_glm(_frequency_data(rng, 50_000), ["region"]), alpha=0.05)
    assert _within_4_se(rel, TRUE_FREQ, alpha=0.05)


def test_true_severity_factors_within_4_se(rng: np.random.Generator) -> None:
    rel = relativities(fit_severity_glm(_severity_data(rng, 20_000), ["region"]), alpha=0.05)
    assert _within_4_se(rel, TRUE_SEV, alpha=0.05)


@pytest.fixture(scope="module")
def portfolio() -> Portfolio:
    # Ein Anfalljahr mit 50 000 Policen: rund 3 600 Schäden.
    return simulate_portfolio(default_parameters(), 50_000, [2024], np.random.default_rng(7))


def _reference(factors: dict[str, dict[str, float]]) -> dict[str, str]:
    return {f: next(lvl for lvl, v in levels.items() if v == 1.0) for f, levels in factors.items()}


@pytest.mark.parametrize("part", ["frequency", "severity"])
def test_simulated_portfolio_recovers_true_factors(portfolio: Portfolio, part: str) -> None:
    params = portfolio.params
    if part == "frequency":
        factors = params.frequency_factors
        data = frequency_table(portfolio.policies, portfolio.claims)
        model = fit_frequency_glm(data, list(factors), _reference(factors))
    else:
        factors = params.severity_factors
        data = severity_table(portfolio.policies, portfolio.claims)
        model = fit_severity_glm(data, list(factors), _reference(factors))

    rel = relativities(model, alpha=0.05)
    for feature, by_level in factors.items():
        assert _within_4_se(rel.loc[[feature]], dict(by_level), alpha=0.05), feature

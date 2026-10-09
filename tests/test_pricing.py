"""Tests für nonlife.pricing.

Zwei Arten von Prüfungen:

* analytisch: Bei einem einzigen kategorialen Merkmal ist das GLM saturiert
  in den Klassen, und der ML-Schätzer hat eine geschlossene Form. Für Poisson
  mit Offset ist exp(β̂_ℓ) = (N_ℓ/E_ℓ) / (N_ref/E_ref), für Gamma mit Log-Link
  ist exp(γ̂_ℓ) = X̄_ℓ / X̄_ref. Toleranz 1e-6 relativ (Abbruchkriterium von IRLS).
* gegen wahre Faktoren: große Stichprobe, das 99-%-Intervall muss den wahren
  Faktor enthalten. Mit festem Seed deterministisch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nonlife.pricing import fit_frequency_glm, fit_severity_glm, relativities
from tests.conftest import not_implemented

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


@not_implemented
def test_poisson_relativities_equal_frequency_ratios(rng: np.random.Generator) -> None:
    data = _frequency_data(rng, 5_000)
    rel = relativities(fit_frequency_glm(data, ["region"]))

    by_region = data.groupby("region")[["n_claims", "exposure"]].sum()
    freq = by_region["n_claims"] / by_region["exposure"]
    for level in ("Stadt", "Vorort"):
        expected = freq[level] / freq["Land"]
        assert rel.loc[("region", level), "factor"] == pytest.approx(expected, rel=1e-6)


@not_implemented
def test_gamma_relativities_equal_mean_ratios(rng: np.random.Generator) -> None:
    data = _severity_data(rng, 3_000)
    rel = relativities(fit_severity_glm(data, ["region"]))

    means = data.groupby("region")["claim_amount"].mean()
    for level in ("Stadt", "Vorort"):
        assert rel.loc[("region", level), "factor"] == pytest.approx(
            means[level] / means["Land"], rel=1e-6
        )


@not_implemented
def test_reference_level_has_factor_one(rng: np.random.Generator) -> None:
    rel = relativities(fit_frequency_glm(_frequency_data(rng, 2_000), ["region"]))
    assert tuple(rel.loc[("region", "Land")]) == (1.0, 1.0, 1.0)
    assert list(rel.columns) == ["factor", "lower", "upper"]


@not_implemented
def test_confidence_interval_contains_estimate(rng: np.random.Generator) -> None:
    rel = relativities(fit_frequency_glm(_frequency_data(rng, 2_000), ["region"]), alpha=0.05)
    assert (rel["lower"] <= rel["factor"]).all()
    assert (rel["factor"] <= rel["upper"]).all()


@not_implemented
def test_wider_interval_for_smaller_alpha(rng: np.random.Generator) -> None:
    result = fit_frequency_glm(_frequency_data(rng, 2_000), ["region"])
    narrow, wide = relativities(result, alpha=0.10), relativities(result, alpha=0.01)
    assert (wide["upper"] >= narrow["upper"]).all()
    assert (wide["lower"] <= narrow["lower"]).all()


@not_implemented
def test_true_frequency_factors_within_ci(rng: np.random.Generator) -> None:
    rel = relativities(fit_frequency_glm(_frequency_data(rng, 50_000), ["region"]), alpha=0.01)
    for level, true in TRUE_FREQ.items():
        row = rel.loc[("region", level)]
        assert row["lower"] <= true <= row["upper"]


@not_implemented
def test_true_severity_factors_within_ci(rng: np.random.Generator) -> None:
    rel = relativities(fit_severity_glm(_severity_data(rng, 20_000), ["region"]), alpha=0.01)
    for level, true in TRUE_SEV.items():
        row = rel.loc[("region", level)]
        assert row["lower"] <= true <= row["upper"]

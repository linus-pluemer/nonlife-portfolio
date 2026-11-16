"""Tests für nonlife.evaluation.

* analytisch: Devianz bei perfekter Vorhersage 0 und bei N = 0 gleich 2μ;
  Ist-Soll-Verhältnis 1 auf den Trainingsdaten (Score-Gleichung des GLM).
* Gegenprobe: Devianz gegen die Implementierung in statsmodels.
* Stichprobe mit festem Seed: Pearson-φ̂ gegen das wahre φ. Ein Summand
  (N − μ)²/μ hat bei Poisson die Varianz 2 + 1/μ, bei μ = 0,1 also 12; mit
  n = 10⁵ ist der Standardfehler von φ̂ etwa √(12/10⁵) ≈ 0,011. Toleranz
  0,05, gut 4 Standardfehler (über 20 Seeds empirisch 0,010). Für die
  Negativ-Binomial-Verteilung ist die Varianz des Summanden größer; über
  20 Seeds liegt der Standardfehler bei 0,019, daher Toleranz 0,08.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from nonlife.evaluation import (
    actual_vs_expected,
    group_split,
    pearson_dispersion,
    poisson_deviance,
)
from nonlife.pricing import fit_frequency_glm, predict_frequency

# --- group_split -------------------------------------------------------------


@pytest.fixture
def groups(rng: np.random.Generator) -> pd.Series:
    sizes = rng.integers(1, 5, size=1_000)  # 1 000 Gruppen mit 1 bis 4 Zeilen
    return pd.Series(np.repeat(np.arange(1_000), sizes))


def test_no_group_on_both_sides(groups: pd.Series, rng: np.random.Generator) -> None:
    test = group_split(groups, 0.1, rng)
    assert not set(groups[test]) & set(groups[~test])


def test_share_of_groups_in_test(groups: pd.Series, rng: np.random.Generator) -> None:
    test = group_split(groups, 0.1, rng)
    assert groups[test].nunique() == 100


def test_same_seed_same_split(groups: pd.Series) -> None:
    first = group_split(groups, 0.1, np.random.default_rng(1))
    second = group_split(groups, 0.1, np.random.default_rng(1))
    assert np.array_equal(first, second)


def test_single_row_groups_split_by_row(rng: np.random.Generator) -> None:
    test = group_split(pd.Series(np.arange(5_000)), 0.1, rng)
    assert test.sum() == 500


def test_invalid_share_rejected(groups: pd.Series, rng: np.random.Generator) -> None:
    with pytest.raises(ValueError, match="test_share"):
        group_split(groups, 1.0, rng)


# --- poisson_deviance --------------------------------------------------------


def test_deviance_zero_for_perfect_prediction() -> None:
    n = np.array([1.0, 2.0, 5.0])
    assert poisson_deviance(n, n) == 0.0


def test_deviance_for_zero_claims_is_two_mu() -> None:
    mu = np.array([0.1, 0.3])
    assert poisson_deviance(np.zeros(2), mu) == pytest.approx(2 * mu.mean(), rel=1e-12)


def test_deviance_matches_statsmodels(rng: np.random.Generator) -> None:
    mu = rng.uniform(0.01, 2.0, size=10_000)
    n = rng.poisson(mu)
    expected = sm.families.Poisson().deviance(n, mu) / len(n)
    assert poisson_deviance(n, mu) == pytest.approx(expected, rel=1e-12)


def _portfolio(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Zwei Merkmale mit multiplikativem Effekt, Exposure in (0,25, 1)."""
    region = rng.choice(["Land", "Stadt"], size=n)
    age = rng.choice(["jung", "mittel", "alt"], size=n)
    exposure = rng.uniform(0.25, 1.0, size=n)
    lam = 0.1 * np.where(region == "Stadt", 1.4, 1.0) * np.where(age == "jung", 1.8, 1.0)
    return pd.DataFrame({"region": region, "age": age, "exposure": exposure,
                         "n_claims": rng.poisson(lam * exposure)})


def test_glm_deviance_not_above_null_model_in_sample(rng: np.random.Generator) -> None:
    # Geschachtelte Modelle: Das größere hat auf den Trainingsdaten nie die größere Devianz.
    data = _portfolio(rng, 20_000)
    glm = fit_frequency_glm(data, ["region", "age"])
    null = fit_frequency_glm(data, [])
    assert poisson_deviance(data["n_claims"], predict_frequency(glm, data)) <= poisson_deviance(
        data["n_claims"], predict_frequency(null, data)
    )


# --- pearson_dispersion ------------------------------------------------------


def test_dispersion_one_for_poisson(rng: np.random.Generator) -> None:
    mu = np.full(100_000, 0.1)
    assert pearson_dispersion(rng.poisson(mu), mu, n_params=1) == pytest.approx(1.0, abs=0.05)


def test_dispersion_negative_binomial(rng: np.random.Generator) -> None:
    # Var[N] = μ + μ²/k, also φ = 1 + μ/k = 1,2 bei μ = 0,1 und k = 0,5.
    mu, k = 0.1, 0.5
    n = rng.negative_binomial(k, k / (k + mu), size=100_000)
    phi = pearson_dispersion(n, np.full(len(n), mu), n_params=1)
    assert phi == pytest.approx(1 + mu / k, abs=0.08)


# --- actual_vs_expected ------------------------------------------------------


def test_actual_vs_expected_sums() -> None:
    data = pd.DataFrame({"region": ["A", "A", "B"], "exposure": [1.0, 0.5, 1.0],
                         "n_claims": [1, 0, 2]})
    table = actual_vs_expected(data, [0.5, 0.25, 1.0], "region")
    assert table.loc["A"].tolist() == [1.5, 1.0, 0.75, pytest.approx(4 / 3)]
    assert table.loc["B", "ratio"] == 2.0


def test_ratio_one_on_training_data(rng: np.random.Generator) -> None:
    # Score-Gleichung Xᵀ(N − μ̂) = 0: Σ N = Σ μ̂ je Ausprägung jedes Merkmals im Modell.
    data = _portfolio(rng, 20_000)
    mu = predict_frequency(fit_frequency_glm(data, ["region", "age"]), data)
    for feature in ("region", "age"):
        ratio = actual_vs_expected(data, mu, feature)["ratio"]
        assert ratio.to_numpy() == pytest.approx(1.0, rel=1e-6)

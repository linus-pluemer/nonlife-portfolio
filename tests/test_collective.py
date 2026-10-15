"""Tests für nonlife.collective.

Toleranzen der Monte-Carlo-Tests: Für n unabhängige Läufe ist der
Standardfehler des Mittels sd(S)/√n. Eine Abweichung von mehr als 4
Standardfehlern hat bei Normalapproximation Wahrscheinlichkeit < 1e-4; mit
festem Seed ist der Test ohnehin deterministisch, die Schranke begründet
nur, warum der gewählte Seed nicht zufällig passend sein muss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nonlife.collective import (
    calculate_statistics,
    discretize_severity,
    estimate_frequency_parameter,
    estimate_severity_parameters,
    monte_carlo_simulation,
    panjer_poisson,
    quantile_from_pmf,
    theoretical_statistics,
)
from tests.conftest import not_implemented

LAMBDA, MU, SIGMA = 50.0, np.log(1500.0), 0.9
N_SIM = 20_000


@pytest.fixture(scope="module")
def severity() -> stats.rv_continuous:
    return stats.lognorm(s=SIGMA, scale=np.exp(MU))


@pytest.fixture(scope="module")
def simulations() -> np.ndarray:
    return monte_carlo_simulation(LAMBDA, MU, SIGMA, N_SIM, np.random.default_rng(1))


# --- Schätzung ---------------------------------------------------------------


def test_frequency_is_claims_per_exposure() -> None:
    policies = pd.DataFrame({"policy_id": [1, 2, 3], "exposure": [1.0, 0.5, 0.5]})
    claims = pd.DataFrame({"policy_id": [1, 1, 3], "claim_amount": [1.0, 2.0, 3.0]})
    assert estimate_frequency_parameter(claims, policies) == pytest.approx(1.5)


def test_severity_fit_recovers_lognormal_parameters(rng: np.random.Generator) -> None:
    # Bei n = 20 000 ist se(σ̂) ≈ σ/√(2n) ≈ 0.0045 und se(μ̂) ≈ σ/√n ≈ 0.0064.
    amounts = rng.lognormal(MU, SIGMA, size=20_000)
    mu, sigma = estimate_severity_parameters(pd.DataFrame({"claim_amount": amounts}))
    assert mu == pytest.approx(MU, abs=0.03)
    assert sigma == pytest.approx(SIGMA, abs=0.02)


# --- Monte Carlo -------------------------------------------------------------


def test_simulation_shape_and_sign(simulations: np.ndarray) -> None:
    assert simulations.shape == (N_SIM,)
    assert (simulations >= 0).all()


def test_simulation_is_reproducible() -> None:
    a = monte_carlo_simulation(LAMBDA, MU, SIGMA, 100, np.random.default_rng(7))
    b = monte_carlo_simulation(LAMBDA, MU, SIGMA, 100, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_zero_frequency_gives_zero_loss(rng: np.random.Generator) -> None:
    assert (monte_carlo_simulation(0.0, MU, SIGMA, 100, rng) == 0).all()


def test_probability_of_no_claim(rng: np.random.Generator) -> None:
    # P(S = 0) = P(N = 0) = e^{-λ}, da X > 0 fast sicher. Prüft, dass Läufe
    # ohne Schaden als 0 erscheinen (minlength in bincount). Toleranz:
    # 4 Standardfehler des Binomialanteils.
    lam, n = 2.0, 20_000
    p = np.exp(-lam)
    share = (monte_carlo_simulation(lam, MU, SIGMA, n, rng) == 0).mean()
    assert abs(share - p) < 4 * np.sqrt(p * (1 - p) / n)


def test_simulated_mean_matches_wald(
    simulations: np.ndarray, severity: stats.rv_continuous
) -> None:
    theory = theoretical_statistics(LAMBDA, severity.mean(), severity.var())
    se_mean = np.sqrt(theory["variance"] / N_SIM)
    assert abs(simulations.mean() - theory["mean"]) < 4 * se_mean


def test_simulated_variance_matches_wald(
    simulations: np.ndarray, severity: stats.rv_continuous
) -> None:
    # Var der Stichprobenvarianz ≈ (μ_4 - σ⁴)/n. Für zusammengesetzt Poisson ist der
    # 4. Kumulant λ·E[X⁴]; μ_4 = κ_4 + 3σ⁴.
    theory = theoretical_statistics(LAMBDA, severity.mean(), severity.var())
    kappa4 = LAMBDA * severity.moment(4)
    se_var = np.sqrt((kappa4 + 2 * theory["variance"] ** 2) / N_SIM)
    assert abs(simulations.var() - theory["variance"]) < 4 * se_var


def test_statistics_keys_and_values() -> None:
    sims = np.arange(101, dtype=float)
    result = calculate_statistics(sims)
    assert result["mean"] == pytest.approx(50.0)
    assert result["variance"] == pytest.approx(np.var(sims))
    assert result["std"] == pytest.approx(np.sqrt(result["variance"]))
    assert result["quantile_99"] == pytest.approx(99.0)


def test_theoretical_statistics_poisson_case() -> None:
    # Für X ≡ c gilt S = c·N, also E[S] = λc und Var[S] = λc².
    result = theoretical_statistics(lambda_=3.0, severity_mean=2.0, severity_variance=0.0)
    assert result["mean"] == pytest.approx(6.0)
    assert result["variance"] == pytest.approx(12.0)


# --- Panjer ------------------------------------------------------------------


@not_implemented
def test_discretized_severity_is_distribution(severity: stats.rv_continuous) -> None:
    pmf = discretize_severity(severity, step=50.0, n_points=2000)
    assert pmf.shape == (2000,)
    assert (pmf >= 0).all()
    assert pmf.sum() == pytest.approx(1.0, abs=1e-12)


@not_implemented
def test_discretized_mean_close_to_true(severity: stats.rv_continuous) -> None:
    # Rundung auf ein Gitter der Weite h verschiebt den Erwartungswert um O(h).
    step = 10.0
    pmf = discretize_severity(severity, step=step, n_points=20_000)
    mean = (np.arange(pmf.size) * step * pmf).sum()
    assert mean == pytest.approx(severity.mean(), abs=step)


@not_implemented
def test_panjer_degenerate_severity_is_poisson() -> None:
    # X ≡ h  ⇒  S/h ~ Poisson(λ): exaktes analytisches Ergebnis.
    severity_pmf = np.array([0.0, 1.0])
    g = panjer_poisson(lambda_=4.0, severity_pmf=severity_pmf, n_points=40)
    np.testing.assert_allclose(g, stats.poisson(4.0).pmf(np.arange(40)), rtol=1e-12, atol=1e-15)


@not_implemented
def test_panjer_mass_at_zero_thinning() -> None:
    # Masse f_0 > 0 verdünnt die Poisson-Anzahl: S ~ Poisson(λ(1 - f_0)) bei X ∈ {0, h}.
    g = panjer_poisson(lambda_=4.0, severity_pmf=np.array([0.25, 0.75]), n_points=40)
    np.testing.assert_allclose(g, stats.poisson(3.0).pmf(np.arange(40)), rtol=1e-12, atol=1e-15)


@not_implemented
def test_panjer_moments_match_wald(severity: stats.rv_continuous) -> None:
    step, n = 50.0, 2000
    f = discretize_severity(severity, step=step, n_points=n)
    g = panjer_poisson(lambda_=LAMBDA, severity_pmf=f, n_points=8000)
    grid = np.arange(f.size) * step
    ex, ex2 = (grid * f).sum(), (grid**2 * f).sum()
    s_grid = np.arange(g.size) * step
    assert g.sum() == pytest.approx(1.0, abs=1e-8)
    assert (s_grid * g).sum() == pytest.approx(LAMBDA * ex, rel=1e-6)
    assert (s_grid**2 * g).sum() - (s_grid * g).sum() ** 2 == pytest.approx(LAMBDA * ex2, rel=1e-6)


@not_implemented
def test_panjer_quantile_close_to_monte_carlo(
    simulations: np.ndarray, severity: stats.rv_continuous
) -> None:
    # Toleranz 3 %: MC-Fehler des 99-%-Quantils bei n = 20 000 plus Diskretisierung.
    step = 50.0
    f = discretize_severity(severity, step=step, n_points=2000)
    g = panjer_poisson(lambda_=LAMBDA, severity_pmf=f, n_points=8000)
    q = quantile_from_pmf(g, step=step, level=0.99)
    assert q == pytest.approx(np.quantile(simulations, 0.99), rel=0.03)

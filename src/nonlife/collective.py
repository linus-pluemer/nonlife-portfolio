"""Kollektives Modell: Frequenz, Schwere und Gesamtschaden.

Der Jahresgesamtschaden ist S = X_1 + ... + X_N mit N ~ Poisson(λ) und
u.i.v. Schadenhöhen X_i ~ Lognormal(μ, σ), unabhängig von N. Es gilt

    E[S]   = E[N]·E[X]
    Var[S] = E[N]·Var[X] + Var[N]·E[X]²

Die Verteilung von S wird per Monte Carlo bestimmt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_frequency_parameter(claims: pd.DataFrame, policies: pd.DataFrame) -> float:
    """Schätzt die Schadenfrequenz je Exposure-Jahr.

    Die Frequenz wird definiert als:
        (Anzahl Schäden) / (Summe Exposure)

    Das ist der ML-Schätzer der Poisson-Rate je Exposure-Jahr, nicht das λ
    des Jahresgesamtschadens. Dieses entsteht erst durch Multiplikation mit
    der Exposure eines Jahres (siehe ``cli._run_collective``).

    Parameters
    ----------
    claims : pd.DataFrame
        DataFrame mit Spalten [policy_id, claim_amount, ...]
    policies : pd.DataFrame
        DataFrame mit Spalten [policy_id, exposure, ...]

    Returns
    -------
    float
        Geschätzte Frequenz je Exposure-Jahr
    """
    return float(len(claims) / policies["exposure"].sum())


def estimate_severity_parameters(claims: pd.DataFrame) -> tuple[float, float]:
    """Schätzt die Parameter einer Lognormal-Verteilung für die Schadenhöhe.

    Mit X ~ Lognormal(μ, σ) ist ln X ~ N(μ, σ²). Die ML-Schätzer sind daher
    Mittelwert und Standardabweichung der logarithmierten Schadenhöhen, die
    Standardabweichung mit Divisor n (ddof=0), wie beim ML-Schätzer der
    Normalverteilung.

    Bewusst nicht ``scipy.stats.lognorm.fit``: ohne ``floc=0`` schätzt es
    zusätzlich eine Verschiebung loc, also ln(X - loc) ~ N(μ, σ²); μ und σ
    gehören dann zu einer anderen Verteilung als der simulierten.

    Parameters
    ----------
    claims : pd.DataFrame
        DataFrame mit Spalte "claim_amount", alle Werte > 0

    Returns
    -------
    (mu, sigma) : tuple[float, float]
        ML-Schätzer der Parameter von ln X ~ N(μ, σ²).
    """
    log_x = np.log(claims["claim_amount"].to_numpy())
    return float(log_x.mean()), float(log_x.std(ddof=0))


def monte_carlo_simulation(
    lambda_: float,
    mu: float,
    sigma: float,
    n_sim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simuliert n_sim Realisierungen von S = X_1 + ... + X_N, vektorisiert.

    Alle Schadenanzahlen N_k werden in einem Aufruf gezogen, danach alle
    Σ N_k Schadenhöhen auf einmal. ``np.bincount`` summiert die Schäden je
    Lauf k; ``minlength`` gibt Läufen ohne Schaden den Wert 0.

    Speicherbedarf O(n_sim · λ): bei λ ≈ 720 und 10 000 Läufen etwa 58 MB.
    Für deutlich mehr Läufe müsste in Blöcken simuliert werden.

    Parameters
    ----------
    lambda_
        Poisson-Parameter, E[N] = λ
    mu, sigma
        Parameter der Lognormalverteilung, also von ln X ~ N(μ, σ²)
    n_sim
        Anzahl Simulationsläufe
    rng
        einzige Zufallsquelle

    Returns
    -------
    Array der Länge n_sim mit den simulierten Gesamtschäden.
    """
    n_claims = rng.poisson(lambda_, size=n_sim)
    amounts = rng.lognormal(mu, sigma, size=n_claims.sum())
    run = np.repeat(np.arange(n_sim), n_claims)
    total: np.ndarray = np.bincount(run, weights=amounts, minlength=n_sim)
    return total


def calculate_statistics(simulations: np.ndarray) -> dict[str, float]:
    """Berechnet Statistiken aus den Simulationsergebnissen.

    Parameters
    ----------
    simulations : np.ndarray
        Array mit Gesamtschäden aus monte_carlo_simulation()

    Returns
    -------
    dict
        Dictionary mit Schlüssel:
        - "mean": Erwartungswert E[S]
        - "variance": Varianz Var[S]
        - "std": Standardabweichung sd[S]
        - "quantile_99": 99-%-Quantil (Value-at-Risk)
    """
    return {
        "mean": float(np.mean(simulations)),
        "variance": float(np.var(simulations, ddof=0)),
        "std": float(np.std(simulations, ddof=0)),
        "quantile_99": float(np.quantile(simulations, 0.99)),
    }


def theoretical_statistics(
    lambda_: float,
    severity_mean: float,
    severity_variance: float,
) -> dict[str, float]:
    """Berechnet Erwartungswert und Varianz von S im kollektiven Modell.

      - E[S] = E[N] * E[X] = λ * E[X]                        (Wald'sche Identität)
      - Var[S] = E[N] * Var[X] + Var[N] * (E[X])^2           (Blackwell-Girshick)
             = λ * Var[X] + λ * (E[X])^2                     (Poisson: Var[N] = λ)

    Parameters
    ----------
    lambda_ : float
        Poisson-Parameter (E[N] = Var[N] = λ)
    severity_mean : float
        Erwartungswert E[X] der Schadenhöhe
    severity_variance : float
        Varianz Var[X] der Schadenhöhe

    Returns
    -------
    dict
        Dictionary mit Schlüssel: "mean", "variance", "std"
    """
    mean = lambda_ * severity_mean
    variance = lambda_ * severity_variance + lambda_ * severity_mean**2
    return {"mean": mean, "variance": variance, "std": float(np.sqrt(variance))}

"""Kollektives Modell: Frequenz, Schwere und Gesamtschaden.

Der Jahresgesamtschaden ist S = X_1 + ... + X_N mit N ~ Poisson(λ) und
u.i.v. Schadenhöhen X_i ~ Lognormal(μ, σ), unabhängig von N. Es gilt

    E[S]   = E[N]·E[X]
    Var[S] = E[N]·Var[X] + Var[N]·E[X]²

Die Verteilung von S wird per Monte Carlo und per Panjer-Rekursion bestimmt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def estimate_frequency_parameter(claims: pd.DataFrame, policies: pd.DataFrame) -> float:
    """Schaetzt den Poisson-Parameter λ der Schadenfrequenz.

    Die Frequenz wird definiert als:
        λ = (Anzahl Schaeden) / (Summe Exposure)

    Dies ist die beobachtete durchschnittliche Schadenzahl pro Einheit Exposure.

    Parameters
    ----------
    claims : pd.DataFrame
        DataFrame mit Spalten [policy_id, claim_amount, ...]
    policies : pd.DataFrame
        DataFrame mit Spalten [policy_id, exposure, ...]

    Returns
    -------
    float
        Geschaetzter λ-Parameter
    """
    return len(claims) / policies["exposure"].sum()
    


def estimate_severity_parameters(claims: pd.DataFrame) -> tuple[float, float]:
    """Schaetzt die Parameter einer Lognormal-Verteilung fuer die Schadenhoehe.

    Die Funktion passt eine Lognormal-Verteilung an die beobachteten
    claim_amounts an und gibt die Parameter μ und σ zurueck.

    Parameters
    ----------
    claims : pd.DataFrame
        DataFrame mit Spalte "claim_amount"

    Returns
    -------
    (mu, sigma) : tuple[float, float]
        Geschaetzte Parameter der Lognormal-Verteilung.
        Hinweis: scipy.stats.lognorm.fit() gibt (s, loc, scale) zurueck,
        wobei s=σ und scale=e^μ.
    """
    sigma, loc, scale = stats.lognorm.fit(claims["claim_amount"])
    mu =  np.log(scale)
    return mu, sigma


def monte_carlo_simulation(
    lambda_: float,
    mu: float,
    sigma: float,
    num_simulations: int = 10000,
    random_seed: int = 42,
) -> np.ndarray:
    """Fuehrt eine Monte-Carlo-Simulation des Gesamtschadens durch.

    Fuer jede Simulation:
      1. Ziehe N ~ Poisson(λ)
      2. Ziehe N Schaeden X_i ~ Lognormal(μ, σ)
      3. Berechne S = sum(X_i)

    Parameters
    ----------
    lambda_ : float
        Poisson-Parameter (Frequenz)
    mu : float
        Lognormal-Parameter μ (log-location)
    sigma : float
        Lognormal-Parameter σ (log-scale)
    num_simulations : int
        Anzahl der Simulationen (default: 10000)
    random_seed : int
        Seed fuer Reproduzierbarkeit (default: 42)

    Returns
    -------
    np.ndarray
        Array mit num_simulations Gesamtschaeden S_1, S_2, ..., S_n
    """
    np.random.seed(random_seed)
    total_losses = []
    for i in range(num_simulations):
        N = np.random.poisson(lambda_)
        if N > 0:
            X = np.random.lognormal(mu, sigma, size=N)
            S = X.sum()
        else:
            S = 0
        total_losses.append(S)
    return np.array(total_losses)


def calculate_statistics(simulations: np.ndarray) -> dict[str, float]:
    """Berechnet Statistiken aus den Simulationsergebnissen.

    Parameters
    ----------
    simulations : np.ndarray
        Array mit Gesamtschaeden aus monte_carlo_simulation()

    Returns
    -------
    dict
        Dictionary mit Schluessel:
        - "mean": Erwartungswert E[S]
        - "variance": Varianz Var[S]
        - "std": Standardabweichung σ[S]
        - "quantile_99": 99%-Quantil (Value-at-Risk)
    """
    
    my_dict = {}
    my_dict["mean"] = np.mean(simulations)
    my_dict["variance"] = np.var(simulations, ddof=0)
    my_dict["std"] = np.std(simulations)
    my_dict["quantile_99"] = np.quantile(simulations, 0.99)
    return my_dict



def theoretical_statistics(
    lambda_: float,
    severity_mean: float,
    severity_variance: float,
) -> dict[str, float]:
    """Berechnet die theoretischen Statistiken nach der Wald'schen Identitaet.

    Wald'sche Identitaet:
      - E[S] = E[N] * E[X] = λ * E[X]
      - Var[S] = E[N] * Var[X] + Var[N] * (E[X])^2
             = λ * Var[X] + λ * (E[X])^2

    Parameters
    ----------
    lambda_ : float
        Poisson-Parameter (E[N] = Var[N] = λ)
    severity_mean : float
        Erwartungswert E[X] der Schadenhoehe
    severity_variance : float
        Varianz Var[X] der Schadenhoehe

    Returns
    -------
    dict
        Dictionary mit Schluessel: "mean", "variance", "std"
    """
    mean = lambda_ * severity_mean
    variance = lambda_ * severity_variance + lambda_ * (severity_mean ** 2)
    std = np.sqrt(variance)
    my_dict = {}
    my_dict["mean"] = mean
    my_dict["variance"] = variance
    my_dict["std"] = std
    return my_dict




# ---------------------------------------------------------------------------
# Panjer-Rekursion -- offen, Implementierung durch Linus
# ---------------------------------------------------------------------------


def discretize_severity(
    severity: stats.rv_continuous, step: float, n_points: int
) -> np.ndarray:
    """Diskretisiert eine stetige Schwereverteilung auf das Gitter {0, h, 2h, ...}.

    Vorschlag: Methode der Rundung (Masse von [kh - h/2, kh + h/2) auf kh,
    Masse von [0, h/2) auf 0). Die Restmasse oberhalb des Gitters wird dem
    letzten Punkt zugeschlagen, damit die Gewichte zu 1 summieren.

    Parameters
    ----------
    severity
        eingefrorene scipy-Verteilung, z. B. ``stats.lognorm(s=sigma, scale=np.exp(mu))``
    step
        Gitterweite h > 0
    n_points
        Anzahl der Gitterpunkte

    Returns
    -------
    np.ndarray der Länge ``n_points`` mit f_k = P(X_disc = k·h), Summe 1.
    """
    raise NotImplementedError


def panjer_poisson(lambda_: float, severity_pmf: np.ndarray, n_points: int) -> np.ndarray:
    """Verteilung des Gesamtschadens per Panjer-Rekursion für N ~ Poisson(λ).

    Poisson gehört zur (a, b, 0)-Klasse mit a = 0, b = λ. Mit f_k = P(X = k·h):

        g_0 = exp(-λ·(1 - f_0))
        g_s = (λ / s) · Σ_{k=1}^{s} k · f_k · g_{s-k},   s ≥ 1

    Parameters
    ----------
    lambda_
        erwartete Schadenanzahl λ > 0
    severity_pmf
        diskretisierte Schwere auf dem Gitter {0, h, 2h, ...}
    n_points
        Länge des Ergebnisgitters für S

    Returns
    -------
    np.ndarray der Länge ``n_points`` mit g_s = P(S = s·h).
    """
    raise NotImplementedError


def quantile_from_pmf(pmf: np.ndarray, step: float, level: float) -> float:
    """Kleinstes s·h mit P(S ≤ s·h) ≥ ``level``."""
    raise NotImplementedError

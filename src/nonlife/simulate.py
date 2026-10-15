"""
Simulation eines synthetischen Kfz-Portfolios mit bekannten wahren Parametern.

Das Modul ist die Datengrundlage des Projekts. Kollektives Modell,
Reservierung und Tarifierung werden gegen die Parameter geprüft, mit denen
hier simuliert wird.

Datenerzeugender Prozess je Anfalljahr und Police i:

1. Tarifmerkmale x_i: ``driver_age`` und ``region`` unabhängig,
   ``vehicle_power`` bedingt auf ``driver_age``. Durch die Abhängigkeit
   sind einfache Auswertungen je Merkmal verzerrt, das GLM nicht.
2. Exposure e_i: mit Wahrscheinlichkeit ``full_year_share`` gleich 1,
   sonst gleichverteilt auf (0, 1].
3. Schadenanzahl N_i ~ Poisson(λ(x_i) · e_i).
4. Schadenhöhe X ~ Lognormal mit E[X | x] laut Parametern und festem
   Variationskoeffizienten c, also σ² = ln(1 + c²), μ = ln E[X | x] - σ²/2.
5. Zahlungsanteile je Schaden ~ Dirichlet(κ · p) mit Abwicklungsmuster p.
   Ohne diesen Zufall hätten alle Anfalljahre identische
   Entwicklungsfaktoren und der Mack-Schätzer σ_j² wäre null.
6. Zahlungen nach dem Bewertungsjahr ``max(accident_years)`` werden
   abgeschnitten.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConditionalDistribution:
    """Verteilung eines Merkmals, bedingt auf ein anderes Merkmal.

    Attributes
    ----------
    given
        Name des bedingenden Merkmals; muss unabhängig gezogen werden
    probabilities
        Ausprägung von ``given`` -> Ausprägung -> Wahrscheinlichkeit
    """

    given: str
    probabilities: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class TrueParameters:
    """Wahre Modellparameter der Simulation, gegen die Schätzungen geprüft werden.

    Frequenz und Schwere sind multiplikativ in den Tarifmerkmalen:

        λ(x) = base_frequency · Π_m frequency_factors[m][x_m]
        E[X | x] = base_severity · Π_m severity_factors[m][x_m]

    Je Merkmal hat genau eine Ausprägung den Faktor 1 (Referenzklasse). Das
    entspricht der Parametrisierung eines GLM mit Log-Link und Treatment-Kodierung.

    Attributes
    ----------
    base_frequency
        erwartete Schadenanzahl je Exposure-Jahr in der Referenzklasse
    frequency_factors
        Merkmal -> Ausprägung -> Frequenzfaktor
    base_severity
        erwartete Schadenhöhe in der Referenzklasse
    severity_factors
        Merkmal -> Ausprägung -> Schwerefaktor
    severity_cv
        Variationskoeffizient der Schadenhöhe, für alle Klassen gleich
        (konstanter CV ist die Annahme des Gamma-GLM)
    development_pattern
        Anteil des Endschadens, der im Abwicklungsjahr j = 0, 1, ... gezahlt
        wird; positiv, Summe 1
    development_concentration
        Konzentration κ der Dirichlet-Verteilung der Zahlungsanteile; je
        größer, desto näher liegen die Anteile am Muster
    feature_probabilities
        unabhängig gezogene Merkmale: Merkmal -> Ausprägung -> Wahrscheinlichkeit
    conditional_features
        bedingt gezogene Merkmale: Merkmal -> bedingte Verteilung
    full_year_share
        Anteil der Policen mit voller Exposure 1
    """

    base_frequency: float
    frequency_factors: Mapping[str, Mapping[str, float]]
    base_severity: float
    severity_factors: Mapping[str, Mapping[str, float]]
    severity_cv: float
    development_pattern: tuple[float, ...]
    development_concentration: float
    feature_probabilities: Mapping[str, Mapping[str, float]]
    conditional_features: Mapping[str, ConditionalDistribution]
    full_year_share: float

    def __post_init__(self) -> None:
        if self.base_frequency <= 0 or self.base_severity <= 0 or self.severity_cv <= 0:
            raise ValueError("base_frequency, base_severity und severity_cv müssen positiv sein")
        if not 0 <= self.full_year_share <= 1:
            raise ValueError("full_year_share muss in [0, 1] liegen")
        if self.development_concentration <= 0:
            raise ValueError("development_concentration muss positiv sein")
        pattern = np.asarray(self.development_pattern)
        if (pattern <= 0).any() or not np.isclose(pattern.sum(), 1.0):
            raise ValueError("development_pattern muss positiv sein und Summe 1 haben")

        features = self.frequency_factors.keys()
        if self.severity_factors.keys() != features:
            raise ValueError("Frequenz und Schwere brauchen dieselben Merkmale")
        if self.feature_probabilities.keys() | self.conditional_features.keys() != features:
            raise ValueError("jedes Merkmal braucht genau eine Verteilung")
        if self.feature_probabilities.keys() & self.conditional_features.keys():
            raise ValueError("ein Merkmal ist zugleich unabhängig und bedingt verteilt")

        for feature in features:
            levels = self.frequency_factors[feature].keys()
            for factors in (self.frequency_factors[feature], self.severity_factors[feature]):
                if factors.keys() != levels:
                    raise ValueError(f"{feature}: Ausprägungen der Faktoren stimmen nicht überein")
                if 1.0 not in factors.values():
                    raise ValueError(f"{feature}: Referenzklasse mit Faktor 1 fehlt")
            if feature in self.feature_probabilities:
                distributions = [self.feature_probabilities[feature]]
            else:
                conditional = self.conditional_features[feature]
                if conditional.given not in self.feature_probabilities:
                    raise ValueError(f"{feature}: bedingendes Merkmal muss unabhängig sein")
                given_levels = self.frequency_factors[conditional.given].keys()
                if conditional.probabilities.keys() != given_levels:
                    raise ValueError(f"{feature}: bedingte Verteilung fehlt für eine Ausprägung")
                distributions = list(conditional.probabilities.values())
            for probabilities in distributions:
                if probabilities.keys() != levels:
                    raise ValueError(f"{feature}: Ausprägungen der Verteilung passen nicht")
                if not np.isclose(sum(probabilities.values()), 1.0):
                    raise ValueError(f"{feature}: Wahrscheinlichkeiten haben nicht Summe 1")


@dataclass(frozen=True)
class Portfolio:
    """Ergebnis von :func:`simulate_portfolio`.

    Attributes
    ----------
    policies
        eine Zeile je Police und Anfalljahr (policy_id je Zeile eindeutig):
        policy_id, accident_year, exposure,
        Tarifmerkmale als Strings
    claims
        eine Zeile je Schaden: claim_id, policy_id, accident_year, claim_amount
        (wahrer Endschaden des Einzelschadens)
    payments
        eine Zeile je Zahlung: claim_id, accident_year, development_year, amount;
        nur Zahlungen bis einschließlich ``valuation_year``
    valuation_year
        Bewertungsstichtag (Ende dieses Kalenderjahres)
    params
        die wahren Parameter, mit denen simuliert wurde
    """

    policies: pd.DataFrame
    claims: pd.DataFrame
    payments: pd.DataFrame
    valuation_year: int
    params: TrueParameters

    def true_ultimate(self) -> pd.Series:
        """Wahrer Endschaden je Anfalljahr (Summe aller claim_amount)."""
        years = np.sort(self.policies["accident_year"].unique())
        return (
            self.claims.groupby("accident_year")["claim_amount"].sum()
            .reindex(years, fill_value=0.0)
            .rename_axis("accident_year")
        )

    def true_reserve(self) -> pd.Series:
        """Wahre Reserve je Anfalljahr: Endschaden minus bis zum Stichtag gezahlt."""
        ultimate = self.true_ultimate()
        paid = self.payments.groupby("accident_year")["amount"].sum()
        return ultimate - paid.reindex(ultimate.index, fill_value=0.0)


def default_parameters() -> TrueParameters:
    """Wahre Parameter für den Standardlauf von CLI und README."""
    return TrueParameters(
        base_frequency=0.07,
        frequency_factors={
            "driver_age": {"18-24": 1.90, "25-39": 1.00, "40-59": 0.85, "60+": 1.10},
            "region": {"urban": 1.30, "suburban": 1.00, "rural": 0.80},
            "vehicle_power": {"low": 1.00, "medium": 1.10, "high": 1.25},
        },
        base_severity=3000.0,
        severity_factors={
            "driver_age": {"18-24": 1.10, "25-39": 1.00, "40-59": 0.97, "60+": 0.95},
            "region": {"urban": 1.15, "suburban": 1.00, "rural": 0.90},
            "vehicle_power": {"low": 1.00, "medium": 1.15, "high": 1.40},
        },
        severity_cv=1.2,
        development_pattern=(0.45, 0.25, 0.12, 0.07, 0.04, 0.03, 0.02, 0.01, 0.007, 0.003),
        development_concentration=20.0,
        feature_probabilities={
            "driver_age": {"18-24": 0.10, "25-39": 0.35, "40-59": 0.35, "60+": 0.20},
            "region": {"urban": 0.35, "suburban": 0.40, "rural": 0.25},
        },
        conditional_features={
            "vehicle_power": ConditionalDistribution(
                given="driver_age",
                probabilities={
                    "18-24": {"low": 0.70, "medium": 0.25, "high": 0.05},
                    "25-39": {"low": 0.40, "medium": 0.40, "high": 0.20},
                    "40-59": {"low": 0.35, "medium": 0.40, "high": 0.25},
                    "60+": {"low": 0.45, "medium": 0.40, "high": 0.15},
                },
            ),
        },
        full_year_share=0.7,
    )


def _draw_categorical(
    levels: Sequence[str], cdf: np.ndarray, rows: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Zieht je Eintrag von ``rows`` eine Ausprägung per Inversionsmethode.

    ``cdf[r]`` ist die Verteilungsfunktion der Ausprägungen für Zeile r;
    die Ausprägung ist die Anzahl der Stufen der Verteilungsfunktion unterhalb von U.
    """
    u = rng.random(len(rows))
    index = (u[:, None] >= cdf[rows]).sum(axis=1)
    drawn: np.ndarray = np.asarray(levels)[index]
    return drawn


def _cdf(probabilities: Sequence[Mapping[str, float]], levels: Sequence[str]) -> np.ndarray:
    cdf: np.ndarray = np.cumsum([[p[level] for level in levels] for p in probabilities], axis=1)
    cdf[:, -1] = 1.0  # Rundungsfehler: U < 1 muss immer eine Ausprägung treffen
    return cdf


def _draw_features(
    params: TrueParameters, n: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    for feature, probabilities in params.feature_probabilities.items():
        levels = list(probabilities)
        features[feature] = _draw_categorical(
            levels, _cdf([probabilities], levels), np.zeros(n, dtype=int), rng
        )
    for feature, conditional in params.conditional_features.items():
        levels = list(params.frequency_factors[feature])
        given_levels = list(conditional.probabilities)
        rows = pd.Categorical(features[conditional.given], categories=given_levels).codes
        cdf = _cdf([conditional.probabilities[g] for g in given_levels], levels)
        features[feature] = _draw_categorical(levels, cdf, np.asarray(rows), rng)
    return features


def _relativity(
    base: float, factors: Mapping[str, Mapping[str, float]], frame: pd.DataFrame
) -> np.ndarray:
    """base · Π_m factors[m][x_m] je Zeile von ``frame``."""
    value = np.full(len(frame), base)
    for feature, by_level in factors.items():
        value *= frame[feature].map(by_level).to_numpy(dtype=float)
    return value


def simulate_portfolio(
    params: TrueParameters,
    n_policies: int,
    accident_years: Sequence[int],
    rng: np.random.Generator,
) -> Portfolio:
    """Simuliert Policen, Einzelschäden und deren Zahlungsverlauf.

    Parameters
    ----------
    params
        wahre Parameter des datenerzeugenden Prozesses
    n_policies
        Anzahl Policen je Anfalljahr; jedes Jahr erhält neue policy_ids
    accident_years
        Anfalljahre; das größte ist das Bewertungsjahr
    rng
        einzige Zufallsquelle

    Returns
    -------
    Portfolio mit Policen, Schäden, Zahlungen bis zum Stichtag und ``params``.
    """
    years = np.asarray(sorted(accident_years), dtype=int)
    if n_policies <= 0 or len(years) == 0:
        raise ValueError("n_policies und accident_years dürfen nicht leer sein")
    valuation_year = int(years[-1])
    n = n_policies * len(years)

    # 1.-2. Policen
    exposure = np.where(rng.random(n) < params.full_year_share, 1.0, 1.0 - rng.random(n))
    policies = pd.DataFrame(
        {
            "policy_id": np.arange(1, n + 1),
            "accident_year": np.repeat(years, n_policies),
            "exposure": exposure,
            **_draw_features(params, n, rng),
        }
    )

    # 3. Schadenanzahl, je Schaden eine Zeile mit Index der zugehörigen Police
    lam = _relativity(params.base_frequency, params.frequency_factors, policies) * exposure
    n_claims = rng.poisson(lam)
    claim_policy = np.repeat(np.arange(n), n_claims)

    # 4. Schadenhöhe
    mean = _relativity(params.base_severity, params.severity_factors, policies)[claim_policy]
    sigma2 = np.log1p(params.severity_cv**2)
    amount = rng.lognormal(mean=np.log(mean) - sigma2 / 2, sigma=np.sqrt(sigma2))
    claims = pd.DataFrame(
        {
            "claim_id": np.arange(1, len(claim_policy) + 1),
            "policy_id": policies["policy_id"].to_numpy()[claim_policy],
            "accident_year": policies["accident_year"].to_numpy()[claim_policy],
            "claim_amount": amount,
        }
    )

    # 5.-6. Zahlungen: Matrix Schaden x Abwicklungsjahr, dann bis zum Stichtag
    alpha = params.development_concentration * np.asarray(params.development_pattern)
    paid = rng.dirichlet(alpha, size=len(claims)) * amount[:, None]
    development = np.arange(len(alpha))
    observed = claims["accident_year"].to_numpy()[:, None] + development <= valuation_year
    claim_index, development_year = np.nonzero(observed)
    payments = pd.DataFrame(
        {
            "claim_id": claims["claim_id"].to_numpy()[claim_index],
            "accident_year": claims["accident_year"].to_numpy()[claim_index],
            "development_year": development_year,
            "amount": paid[claim_index, development_year],
        }
    )

    return Portfolio(policies, claims, payments, valuation_year, params)

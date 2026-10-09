"""
Simulation eines synthetischen Schadendatensatzes (Kfz-Haftpflicht, Nichtleben).

Dieses Modul ist die Datengrundlage für das gesamte Projekt. Es erzeugt
ein fiktives Portfolio von Kfz-Policen samt Schäden, an dem kollektives
Modell, Reservierung und Tarifierung gegen bekannte wahre Parameter
geprüft werden.

Das Modul enthält zwei Stände:

* ``simulate_policies``/``simulate_claims``: bisherige Simulation ohne
  Anfall- und Meldejahr. Sie bleibt, bis die neue Schnittstelle fertig ist.
* ``TrueParameters``/``simulate_portfolio``: Entwurf der neuen
  Schnittstelle mit kategorialen Merkmalen, mehrjähriger Abwicklung und
  den wahren Parametern als Rückgabe. Die Implementierung ist offen.

Modellidee der bisherigen Simulation:

* Jede Police hat eine Exposure (Anteil des Jahres, den die Police aktiv
  war), sowie ein paar Tarifmerkmale: Fahreralter, PS-Klasse, Region.
* Die Schadenanzahl pro Police wird ueber eine Poisson-Verteilung gezogen.
  Die Rate (lambda) haengt multiplikativ von den Tarifmerkmalen ab -- das
  ist genau die Struktur, die das Frequenz-GLM wieder "entdecken" soll.
* Jede einzelne Schadenhoehe wird ueber eine Lognormalverteilung gezogen.

Alles ist mit einem festen Zufalls-Seed reproduzierbar.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Tarif-Faktoren fuer die Schadenfrequenz (bewusst simpel, multiplikativ)
# ---------------------------------------------------------------------------

BASE_FREQUENCY = 0.08  # erwartete Schadenanzahl pro Jahr bei "Normalrisiko"


def _alters_faktor(alter: np.ndarray) -> np.ndarray:
    """Junge und sehr alte Fahrer:innen haben ein hoeheres Risiko (U-Form)."""
    return 1.0 + 1.5 * np.exp(-((alter - 25) / 12) ** 2) + 0.3 * np.exp(-((alter - 70) / 15) ** 2)


def _region_faktor(region: np.ndarray) -> np.ndarray:
    mapping = {"Stadt": 1.4, "Vorort": 1.1, "Land": 0.8}
    return np.array([mapping[r] for r in region])


def _ps_faktor(ps_klasse: np.ndarray) -> np.ndarray:
    # ps_klasse liegt zwischen 1 (schwach motorisiert) und 5 (stark)
    return 0.8 + 0.15 * ps_klasse


def simulate_policies(n_policies: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Erzeugt ein Portfolio von Kfz-Policen mit Tarifmerkmalen.

    Returns
    -------
    DataFrame mit einer Zeile pro Police:
        policy_id, exposure, fahrer_alter, ps_klasse, region, erwartete_frequenz
    """
    rng = np.random.default_rng(seed)

    policy_id = np.arange(1, n_policies + 1)
    exposure = np.round(rng.uniform(0.25, 1.0, n_policies), 2)
    fahrer_alter = rng.integers(18, 80, n_policies)
    ps_klasse = rng.integers(1, 6, n_policies)
    region = rng.choice(["Stadt", "Vorort", "Land"], size=n_policies, p=[0.4, 0.35, 0.25])

    lam = (
        BASE_FREQUENCY
        * _alters_faktor(fahrer_alter)
        * _region_faktor(region)
        * _ps_faktor(ps_klasse)
        * exposure
    )

    return pd.DataFrame(
        {
            "policy_id": policy_id,
            "exposure": exposure,
            "fahrer_alter": fahrer_alter,
            "ps_klasse": ps_klasse,
            "region": region,
            "erwartete_frequenz": lam,
        }
    )


def simulate_claims(policies: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Zieht fuer jede Police eine Schadenanzahl und je Schaden eine Schadenhoehe.

    Returns
    -------
    DataFrame mit einer Zeile pro EINZELSCHADEN:
        claim_id, policy_id, claim_amount
    """
    rng = np.random.default_rng(seed + 1)  # anderer Seed als die Policen

    n_claims_per_policy = rng.poisson(policies["erwartete_frequenz"].to_numpy())

    rows = []
    claim_id = 1
    # Schadenhoehe: Lognormalverteilung, Median ~ 1500 EUR, langer rechter Schwanz
    mu, sigma = np.log(1500), 0.9

    for policy_id, n_claims in zip(policies["policy_id"], n_claims_per_policy):
        if n_claims == 0:
            continue
        amounts = rng.lognormal(mean=mu, sigma=sigma, size=n_claims)
        for amount in amounts:
            rows.append({"claim_id": claim_id, "policy_id": policy_id, "claim_amount": round(float(amount), 2)})
            claim_id += 1

    return pd.DataFrame(rows, columns=["claim_id", "policy_id", "claim_amount"])


# ---------------------------------------------------------------------------
# Neue Schnittstelle -- Entwurf, Implementierung durch Linus
# ---------------------------------------------------------------------------


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
        wird; nichtnegativ, Summe 1
    """

    base_frequency: float
    frequency_factors: Mapping[str, Mapping[str, float]]
    base_severity: float
    severity_factors: Mapping[str, Mapping[str, float]]
    severity_cv: float
    development_pattern: tuple[float, ...]


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
        raise NotImplementedError


def default_parameters() -> TrueParameters:
    """Wahre Parameter für den Standardlauf von CLI und README."""
    raise NotImplementedError


def simulate_portfolio(
    params: TrueParameters,
    n_policies: int,
    accident_years: Sequence[int],
    rng: np.random.Generator,
) -> Portfolio:
    """Simuliert Policen, Einzelschäden und deren Zahlungsverlauf.

    Vorgehen (Vorschlag):

    1. Für jede Police und jedes Anfalljahr Merkmale und Exposure ziehen.
    2. N ~ Poisson(λ(x) · exposure), vektorisiert über alle Policen.
    3. Je Schaden X mit E[X | x] laut ``params`` und Variationskoeffizient
       ``params.severity_cv``, z. B. Gamma oder Lognormal.
    4. Jeden Schaden nach ``params.development_pattern`` auf die
       Abwicklungsjahre verteilen (deterministisch oder zufällig, z. B.
       multinomial bzw. Dirichlet um das Muster).
    5. Zahlungen nach ``max(accident_years)`` abschneiden.

    Zufall ausschließlich über ``rng``. Keine Python-Schleifen über Policen
    oder Schäden.
    """
    raise NotImplementedError

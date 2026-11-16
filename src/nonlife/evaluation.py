"""Bewertung eines Frequenzmodells ohne wahre Parameter.

Auf echten Daten gibt es keine wahren Faktoren zum Vergleich. Gemessen wird
deshalb die Vorhersagegüte auf Policen, die das Modell nicht gesehen hat:

* :func:`group_split` teilt in Trainings- und Testdaten, ganze Gruppen auf
  einmal, damit mehrfach erfasste Verträge nicht auf beiden Seiten landen.
* :func:`poisson_deviance` ist das Maß der Vorhersagegüte, dasselbe
  Kriterium, das das Poisson-GLM bei der Schätzung minimiert.
* :func:`pearson_dispersion` prüft die Poisson-Annahme Var[N] = E[N].
* :func:`actual_vs_expected` vergleicht beobachtete und vorhergesagte
  Schadenzahlen je Klasse eines Merkmals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.special import xlogy


def group_split(groups: pd.Series, test_share: float, rng: np.random.Generator) -> np.ndarray:
    """Boolesche Maske „im Test“ je Zeile; jede Gruppe liegt ganz auf einer Seite.

    Zieht round(test_share · G) der G Gruppen ohne Zurücklegen. Der Anteil
    der Zeilen im Test liegt dann nur ungefähr bei ``test_share``, bei
    kleinen Gruppen aber sehr nah daran. Für eine Aufteilung je Zeile jeder
    Zeile eine eigene Gruppe geben.

    Parameters
    ----------
    groups
        Gruppenkennung je Zeile, etwa aus :func:`nonlife.fremtpl2.duplicate_groups`
    test_share
        Anteil der Gruppen im Test, in (0, 1)
    rng
        Zufallsgenerator; gleicher Seed, gleiche Aufteilung
    """
    if not 0 < test_share < 1:
        raise ValueError("test_share muss in (0, 1) liegen")
    unique = groups.unique()
    n_test = round(test_share * len(unique))
    test_groups = rng.choice(unique, size=n_test, replace=False)
    return np.asarray(groups.isin(test_groups).to_numpy(bool))


def poisson_deviance(n: ArrayLike, mu: ArrayLike) -> float:
    """Mittlere Poisson-Devianz (1/n) Σ 2 [N_i log(N_i / μ_i) − (N_i − μ_i)].

    Das ist das Doppelte der Differenz der Log-Likelihoods von saturiertem
    Modell (μ_i = N_i) und geschätztem Modell, je Beobachtung. Für N_i = 0
    gilt 0 · log 0 = 0, der Summand ist dann 2 μ_i; ``xlogy`` setzt das um.
    Kleiner ist besser, 0 nur bei perfekter Vorhersage.
    """
    n_arr = np.asarray(n, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    unit = 2 * (xlogy(n_arr, n_arr) - xlogy(n_arr, mu_arr) - (n_arr - mu_arr))
    return float(unit.mean())


def pearson_dispersion(n: ArrayLike, mu: ArrayLike, n_params: int) -> float:
    """Pearson-Schätzer der Dispersion φ̂ = Σ (N_i − μ_i)² / μ_i / (n − p).

    Unter der Poisson-Annahme ist φ = 1. Deutlich größere Werte zeigen
    Überdispersion: Die Schätzer bleiben konsistent, aber Standardfehler
    mit φ = 1 sind um den Faktor √φ zu klein.
    """
    n_arr = np.asarray(n, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    return float(((n_arr - mu_arr) ** 2 / mu_arr).sum() / (len(n_arr) - n_params))


def actual_vs_expected(
    data: pd.DataFrame,
    mu: ArrayLike,
    feature: str,
    claim_count: str = "n_claims",
    exposure: str = "exposure",
) -> pd.DataFrame:
    """Beobachtete gegen vorhergesagte Schadenzahl je Ausprägung von ``feature``.

    Auf den Trainingsdaten ist ``ratio`` für jedes Merkmal im Modell genau 1:
    Das ist die Score-Gleichung Xᵀ(N − μ̂) = 0 des Poisson-GLM mit Log-Link.
    Aussagekräftig ist der Vergleich deshalb auf den Testdaten oder für
    Merkmale, die nicht im Modell sind.

    Returns
    -------
    DataFrame mit Index = Ausprägung und den Spalten exposure, actual
    (Σ N), expected (Σ μ̂) und ratio = actual / expected.
    """
    frame = pd.DataFrame(
        {
            "level": data[feature].array,  # .array behält eine geordnete Kategorie bei
            "exposure": data[exposure].to_numpy(float),
            "actual": data[claim_count].to_numpy(float),
            "expected": np.asarray(mu, dtype=float),
        }
    )
    table = frame.groupby("level", observed=True)[["exposure", "actual", "expected"]].sum()
    table["ratio"] = table["actual"] / table["expected"]
    return table.rename_axis(feature)

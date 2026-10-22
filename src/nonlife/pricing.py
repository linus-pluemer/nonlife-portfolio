"""Tarifierung: Poisson-GLM für die Frequenz, Gamma-GLM für die Schwere.

Frequenz:  log E[N_i] = log(e_i) + β_0 + Σ_m β_{m, x_im}   (Exposure e_i als Offset)
Schwere:   log E[X_k] = γ_0 + Σ_m γ_{m, x_km}

Mit Treatment-Kodierung sind exp(β_{m,ℓ}) und exp(γ_{m,ℓ}) die Tariffaktoren
der Ausprägung ℓ relativ zur Referenzausprägung des Merkmals m.

Entwurfsentscheidungen:

* Die Designmatrix wird hier selbst gebaut (eine 0/1-Spalte je Ausprägung
  außer der Referenz) statt über die Formel-Schnittstelle von statsmodels.
  Die Referenz ist frei wählbar, und ``relativities`` muss keine
  Parameternamen der Formelbibliothek parsen.
* Ergebnis ist ein :class:`TariffModel`, das neben dem statsmodels-Ergebnis
  die Ausprägungen und Referenzen trägt, die zur Auswertung nötig sind.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from nonlife.data import observed_frequency_by

GLMResults = Any  # statsmodels.genmod.generalized_linear_model.GLMResultsWrapper


@dataclass(frozen=True)
class TariffModel:
    """Angepasstes GLM mit den Angaben, die zur Auswertung nötig sind.

    Attributes
    ----------
    result
        statsmodels-Ergebnis
    levels
        Merkmal -> alle Ausprägungen in sortierter Reihenfolge
    reference
        Merkmal -> Referenzausprägung (Faktor 1)
    """

    result: GLMResults
    levels: Mapping[str, tuple[Hashable, ...]]
    reference: Mapping[str, Hashable]


# --- Datenaufbereitung -------------------------------------------------------


def frequency_table(policies: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    """Eine Zeile je Police mit der Schadenanzahl ``n_claims`` (0, wenn kein Schaden)."""
    counts = claims["policy_id"].value_counts()
    table = policies.copy()
    table["n_claims"] = policies["policy_id"].map(counts).fillna(0).astype(int)
    return table


def severity_table(policies: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    """Eine Zeile je Schaden mit ``claim_amount`` und den Merkmalen der Police.

    Spalten, die in beiden Tabellen stehen (etwa ``accident_year``), werden aus
    ``claims`` übernommen. ``validate`` stellt sicher, dass jede policy_id in
    ``policies`` eindeutig ist, sonst würden Schäden beim Merge verdoppelt.
    """
    shared = [c for c in policies.columns if c in claims.columns and c != "policy_id"]
    return claims.merge(
        policies.drop(columns=shared), on="policy_id", how="left", validate="many_to_one"
    )


# --- Schätzung ---------------------------------------------------------------


def _column(feature: str, level: Hashable) -> str:
    return f"{feature}[{level}]"


def _design_matrix(
    data: pd.DataFrame, features: Sequence[str], reference: Mapping[str, Hashable] | None
) -> tuple[pd.DataFrame, dict[str, tuple[Hashable, ...]], dict[str, Hashable]]:
    """Achsenabschnitt plus eine 0/1-Spalte je Ausprägung außer der Referenz.

    Ohne Angabe in ``reference`` ist die erste Ausprägung in sortierter
    Reihenfolge die Referenz.
    """
    columns: dict[str, np.ndarray] = {"const": np.ones(len(data))}
    levels: dict[str, tuple[Hashable, ...]] = {}
    ref: dict[str, Hashable] = {}
    for feature in features:
        levels[feature] = tuple(sorted(data[feature].unique()))
        ref[feature] = (reference or {}).get(feature, levels[feature][0])
        if ref[feature] not in levels[feature]:
            raise ValueError(f"{feature}: Referenz {ref[feature]!r} kommt in den Daten nicht vor")
        for level in levels[feature]:
            if level != ref[feature]:
                columns[_column(feature, level)] = (data[feature] == level).to_numpy(float)
    return pd.DataFrame(columns, index=data.index), levels, ref


def fit_frequency_glm(
    data: pd.DataFrame,
    features: Sequence[str],
    reference: Mapping[str, Hashable] | None = None,
    claim_count: str = "n_claims",
    exposure: str = "exposure",
) -> TariffModel:
    """Schätzt das Poisson-GLM mit Log-Link und log(Exposure) als Offset.

    Der Offset hat den festen Koeffizienten 1, weil E[N_i] proportional zur
    Exposure angenommen wird. Die Dispersion ist auf 1 festgelegt
    (``scale=1.0``, reines Poisson-Modell).

    Parameters
    ----------
    data
        eine Zeile je Police mit Schadenanzahl, Exposure und Merkmalen
        (siehe :func:`frequency_table`)
    features
        kategoriale Merkmale
    reference
        Merkmal -> Referenzausprägung; fehlende Merkmale: erste Ausprägung
        in sortierter Reihenfolge
    claim_count, exposure
        Spaltennamen
    """
    design, levels, ref = _design_matrix(data, features, reference)
    model = sm.GLM(
        data[claim_count],
        design,
        family=sm.families.Poisson(),
        offset=np.log(data[exposure].to_numpy(float)),
    )
    return TariffModel(model.fit(scale=1.0), levels, ref)


def fit_severity_glm(
    data: pd.DataFrame,
    features: Sequence[str],
    reference: Mapping[str, Hashable] | None = None,
    amount: str = "claim_amount",
) -> TariffModel:
    """Schätzt das Gamma-GLM mit Log-Link für die Einzelschadenhöhe.

    Das Gamma-GLM unterstellt Var[X] = φ·μ², also einen festen
    Variationskoeffizienten. Für konsistente Schätzer reicht auch ohne
    Gamma-Verteilung eine korrekt spezifizierte Erwartungsstruktur
    (Quasi-Likelihood); die Varianzfunktion wird erst für gültige
    Standardfehler gebraucht. Beides trifft auf die lognormalen Schäden der
    Simulation mit festem σ zu. Die Dispersion φ wird über die
    Pearson-Statistik geschätzt (``scale="X2"``).

    Parameters
    ----------
    data
        eine Zeile je Schaden mit Schadenhöhe und Merkmalen der Police
        (siehe :func:`severity_table`)
    features, reference
        wie bei :func:`fit_frequency_glm`
    amount
        Spaltenname der Schadenhöhe
    """
    design, levels, ref = _design_matrix(data, features, reference)
    model = sm.GLM(
        data[amount], design, family=sm.families.Gamma(link=sm.families.links.Log())
    )
    return TariffModel(model.fit(scale="X2"), levels, ref)


# --- Auswertung --------------------------------------------------------------


def relativities(model: TariffModel, alpha: float = 0.05) -> pd.DataFrame:
    """Tariffaktoren exp(β) mit (1 - alpha)-Konfidenzintervall je Merkmalsausprägung.

    Das Intervall ist exp(β̂ ± z_{1-α/2} · se(β̂)) (Wald-Intervall auf der
    Log-Skala, daher asymmetrisch um den Faktor).

    Returns
    -------
    DataFrame mit MultiIndex (feature, level) und den Spalten factor, lower,
    upper. Die Referenzausprägung jedes Merkmals ist enthalten, mit
    factor = lower = upper = 1. Der Achsenabschnitt ist nicht enthalten.
    """
    z = stats.norm.ppf(1 - alpha / 2)
    params, se = model.result.params, model.result.bse
    rows = []
    for feature, levels in model.levels.items():
        for level in levels:
            if level == model.reference[feature]:
                rows.append((feature, level, 1.0, 1.0, 1.0))
                continue
            beta, s = params[_column(feature, level)], se[_column(feature, level)]
            rows.append(
                (feature, level, np.exp(beta), np.exp(beta - z * s), np.exp(beta + z * s))
            )
    frame = pd.DataFrame(rows, columns=["feature", "level", "factor", "lower", "upper"])
    return frame.set_index(["feature", "level"])


def one_way_relativities(
    policies: pd.DataFrame, claims: pd.DataFrame, feature: str, reference: Hashable
) -> pd.Series:
    """Einfache Frequenzrelativität je Ausprägung: (N_ℓ/E_ℓ) / (N_ref/E_ref).

    Entspricht dem Poisson-GLM mit nur diesem einen Merkmal. Sind die
    Merkmale korreliert, nimmt der Faktor die Effekte der übrigen Merkmale
    mit auf; der Vergleich mit dem GLM zeigt diese Verzerrung.
    """
    frequency = observed_frequency_by(policies, claims, feature).set_index(feature)["frequency"]
    return (frequency / frequency[reference]).rename("factor")

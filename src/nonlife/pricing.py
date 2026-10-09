"""Tarifierung: Poisson-GLM für die Frequenz, Gamma-GLM für die Schwere.

Frequenz:  log E[N_i] = log(e_i) + β_0 + Σ_m β_{m, x_im}   (Exposure e_i als Offset)
Schwere:   log E[X_k] = γ_0 + Σ_m γ_{m, x_km}

Mit Treatment-Kodierung sind exp(β_{m,ℓ}) und exp(γ_{m,ℓ}) die Tariffaktoren
der Ausprägung ℓ relativ zur Referenzausprägung des Merkmals m.

Implementierung offen (Linus).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

GLMResults = Any  # statsmodels.genmod.generalized_linear_model.GLMResultsWrapper


def fit_frequency_glm(
    data: pd.DataFrame,
    features: Sequence[str],
    claim_count: str = "n_claims",
    exposure: str = "exposure",
) -> GLMResults:
    """Schätzt das Poisson-GLM mit Log-Link und log(Exposure) als Offset.

    Parameters
    ----------
    data
        eine Zeile je Police mit Schadenanzahl, Exposure und Merkmalen
    features
        kategoriale Merkmale; Referenzklasse ist jeweils die erste
        Ausprägung in sortierter Reihenfolge (Standard von ``C()`` in der
        Formel-Schnittstelle von statsmodels)
    claim_count, exposure
        Spaltennamen

    Returns
    -------
    angepasstes statsmodels-Ergebnisobjekt
    """
    raise NotImplementedError


def fit_severity_glm(
    data: pd.DataFrame,
    features: Sequence[str],
    amount: str = "claim_amount",
) -> GLMResults:
    """Schätzt das Gamma-GLM mit Log-Link für die Einzelschadenhöhe.

    Parameters
    ----------
    data
        eine Zeile je Schaden mit Schadenhöhe und Merkmalen der Police
    features, amount
        wie bei :func:`fit_frequency_glm`

    Returns
    -------
    angepasstes statsmodels-Ergebnisobjekt
    """
    raise NotImplementedError


def relativities(result: GLMResults, alpha: float = 0.05) -> pd.DataFrame:
    """Tariffaktoren exp(β) mit (1 - alpha)-Konfidenzintervall je Merkmalsausprägung.

    Das Intervall ist exp(β̂ ± z_{1-α/2} · se(β̂)) (Wald-Intervall auf der
    Log-Skala, daher asymmetrisch um den Faktor).

    Returns
    -------
    DataFrame mit MultiIndex (feature, level) und den Spalten factor, lower,
    upper. Die Referenzausprägung jedes Merkmals ist enthalten, mit
    factor = lower = upper = 1. Der Achsenabschnitt ist nicht enthalten.
    """
    raise NotImplementedError

"""Reservierung: Zahlungsdreieck und Chain-Ladder.

Bezeichnungen: C_{i,j} ist der kumulierte Zahlbetrag des Anfalljahres i
nach j Abwicklungsjahren. Die volumengewichteten Entwicklungsfaktoren sind

    f_j = Σ_i C_{i,j+1} / Σ_i C_{i,j}

summiert über alle Anfalljahre, für die beide Werte beobachtet sind. Der
Endschaden wird als C_{i,J} = C_{i,I-i} · Π_{j ≥ I-i} f_j hochgerechnet, die
Reserve ist C_{i,J} - C_{i,I-i}.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def create_claims_triangle(payments: pd.DataFrame, valuation_year: int) -> pd.DataFrame:
    """Kumuliertes Zahlungsdreieck aus Einzelzahlungen.

    Zelle (i, j) enthält C_{i,j} = Σ der Zahlungen des Anfalljahres i in den
    Abwicklungsjahren 0..j. Beobachtet ist die Zelle genau dann, wenn
    i + j ≤ valuation_year. Hat eine beobachtete Zelle keine Zahlung, ist der
    Zuwachs 0 und C_{i,j} = C_{i,j-1}. Zellen nach dem Stichtag sind NaN.

    Der Stichtag wird übergeben und nicht aus den Daten abgeleitet: Hätte das
    jüngste Anfalljahr noch keine Zahlung, wäre max(accident_year) stillschweigend
    falsch. Das Dreieck ist quadratisch; Abwicklungsjahre ohne Zahlungen ergeben
    Faktoren f_j = 1.

    Parameters
    ----------
    payments
        Spalten accident_year, development_year, amount
    valuation_year
        Bewertungsstichtag (Ende des Kalenderjahres)

    Returns
    -------
    DataFrame mit Index accident_year (alle Jahre von min(accident_year) bis
    valuation_year) und Spalten development_year 0..valuation_year − min(accident_year).
    """
    if payments.empty:
        raise ValueError("payments ist leer")
    if (payments["development_year"] < 0).any():
        raise ValueError("development_year darf nicht negativ sein")
    if (payments["accident_year"] + payments["development_year"] > valuation_year).any():
        raise ValueError(f"payments enthält Zahlungen nach dem Stichtag {valuation_year}")

    first_year = int(payments["accident_year"].min())
    accident_years = pd.RangeIndex(first_year, valuation_year + 1, name="accident_year")
    development_years = pd.RangeIndex(0, valuation_year - first_year + 1, name="development_year")

    # Zuwächse auf dem vollen Gitter; beobachtete Zellen ohne Zahlung erhalten 0
    incremental = payments.pivot_table(
        index="accident_year", columns="development_year", values="amount",
        aggfunc="sum", fill_value=0.0,
    ).reindex(index=accident_years, columns=development_years, fill_value=0.0)

    # Kumulieren, danach die Zellen jenseits der Kalenderjahrdiagonale ausblenden
    after_valuation = (
        accident_years.to_numpy()[:, None] + development_years.to_numpy() > valuation_year
    )
    return incremental.cumsum(axis=1).mask(after_valuation)


def calculate_development_factors(triangle: pd.DataFrame) -> np.ndarray:
    """Berechnet Entwicklungsfaktoren fuer das Chain-Ladder-Verfahren.

    Entwicklungsfaktor fuer Jahr j:
      f_j = sum(Spalte j+1) / sum(Spalte j)
      (nur ueber Zeilen, in denen beide Werte beobachtet sind)

    Das sagt aus: "Im Durchschnitt multiplizieren sich die Schaeden
    vom Jahr j zum Jahr j+1 mit diesem Faktor."

    Returns: Array der Entwicklungsfaktoren
    """
    columns = triangle.columns.tolist()
    factors = []

    for i in range(len(columns) - 1):
        col_j = columns[i]
        col_j1 = columns[i + 1]

        # Nur Zeilen verwenden, in denen beide Spalten beobachtet sind
        both = triangle[col_j].notna() & triangle[col_j1].notna()
        factor = triangle.loc[both, col_j1].sum() / triangle.loc[both, col_j].sum()
        factors.append(factor)

    return np.array(factors)


def project_triangle(triangle: pd.DataFrame, factors: np.ndarray) -> pd.DataFrame:
    """Projiziert das Schadendreieck in die Zukunft mit Chain-Ladder.

    Fuelle die NaN-Werte (rechts unterhalb der Diagonale) auf, indem du
    innerhalb jeder Zeile den Wert der vorherigen Spalte mit dem
    entsprechenden Entwicklungsfaktor multiplizierst.

    Returns: Vollstaendige (projizierte) Tabelle
    """
    projected = triangle.copy()

    # Iteriere durch die Spalten und fülle fehlende Werte aus
    columns = projected.columns.tolist()
    for col_idx in range(1, len(columns)):
        prev_col = columns[col_idx - 1]
        curr_col = columns[col_idx]
        factor = factors[col_idx - 1]

        # Fülle NaN-Werte mit Wert_vorherige_Spalte * Faktor
        mask = projected[curr_col].isna()
        projected.loc[mask, curr_col] = projected.loc[mask, prev_col] * factor

    return projected


def calculate_reserve(
    original_triangle: pd.DataFrame, projected_triangle: pd.DataFrame
) -> dict[str, Any]:
    """Berechnet die Reserve (IBNR) pro Anfalljahr und insgesamt.

    Reserve fuer ein Anfalljahr = projizierter Gesamtschaden - beobachteter Gesamtschaden

    Returns: Dictionary mit 'reserve_by_year', 'total_reserve', und ggf. 'projected_total'
    """
    # Letzter beobachteter kumulierter Wert pro Anfalljahr (die aktuelle Diagonale)
    latest_observed = original_triangle.ffill(axis=1).iloc[:, -1]

    # Endschaden (Ultimate) = letzte Spalte des projizierten Dreiecks
    ultimate = projected_triangle.iloc[:, -1]

    reserve_by_year = ultimate - latest_observed

    return {
        'reserve_by_year': reserve_by_year,
        'total_reserve': reserve_by_year.sum(),
        'projected_total': ultimate.sum()
    }

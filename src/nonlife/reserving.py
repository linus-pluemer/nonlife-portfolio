"""Reservierung: Schadendreieck, Chain-Ladder und Standardfehler nach Mack.

Bezeichnungen: C_{i,j} ist der kumulierte Schadenaufwand des Anfalljahres i
nach j Abwicklungsjahren. Die volumengewichteten Entwicklungsfaktoren sind

    f_j = Σ_i C_{i,j+1} / Σ_i C_{i,j}

summiert über alle Anfalljahre, für die beide Werte beobachtet sind. Der
Endschaden wird als C_{i,J} = C_{i,I-i} · Π_{j ≥ I-i} f_j hochgerechnet, die
Reserve ist C_{i,J} - C_{i,I-i}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_claims_triangle(claims: pd.DataFrame) -> pd.DataFrame:
    """Erstellt ein kumuliertes Schadendreieck aus den Einzelschadendaten.

    Das Dreieck soll folgende Struktur haben:
      - Index: Anfalljahr (accident_year)
      - Spalten: Abwicklungsjahr (development_year)
      - Werte: kumulierte Schadensummen

    Hinweis:
      - Berechne das Abwicklungsjahr aus den Zeitstempeln (wann wurde der Schaden
        gemeldet vs. wann trat er ein)
      - Nutze groupby + sum, um kumulierte Schaeden zu berechnen
      - Achte auf NaN-Werte fuer noch nicht abgewickelte Schaeden
    """
    # Gruppiere nach (accident_year, development_year) und summiere claim_amount
    grouped = claims.groupby(['accident_year', 'development_year'])['claim_amount'].sum()

    # Erstelle ein Pivot-Table mit accident_year als Index und development_year als Spalten
    triangle = grouped.unstack(fill_value=np.nan)

    # Sortiere nach accident_year (Index) und development_year (Spalten)
    triangle = triangle.sort_index()
    triangle = triangle.reindex(sorted(triangle.columns), axis=1)

    # Inkrementelle Werte zu kumulierten Werten aufsummieren (NaN bleiben NaN)
    triangle = triangle.cumsum(axis=1)

    return triangle


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


def calculate_reserve(original_triangle: pd.DataFrame, projected_triangle: pd.DataFrame) -> dict:
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




# ---------------------------------------------------------------------------
# Standardfehler nach Mack -- offen, Implementierung durch Linus
# ---------------------------------------------------------------------------


def mack_sigma2(triangle: pd.DataFrame, factors: np.ndarray) -> np.ndarray:
    """Schätzer der Varianzparameter σ_j² im Modell von Mack (1993).

        σ_j² = 1/(n_j - 1) · Σ_i C_{i,j} · (C_{i,j+1}/C_{i,j} - f_j)²

    mit n_j der Anzahl beobachteter Paare. Für das letzte j (n_j = 1) die
    Extrapolation von Mack: σ_{J-1}² = min(σ_{J-2}⁴ / σ_{J-3}², σ_{J-3}², σ_{J-2}²).

    Returns
    -------
    np.ndarray der Länge ``len(factors)``.
    """
    raise NotImplementedError


def mack_standard_error(triangle: pd.DataFrame, factors: np.ndarray) -> pd.DataFrame:
    """Standardfehler der Chain-Ladder-Reserve je Anfalljahr und gesamt nach Mack (1993).

    msep(R̂_i) = Ĉ_{i,J}² · Σ_{j=I-i}^{J-1} σ_j²/f_j² · (1/Ĉ_{i,j} + 1/Σ_k C_{k,j})

    mit Σ_k über alle Anfalljahre k, für die C_{k,j+1} beobachtet ist.
    Für die Gesamtreserve kommen die Kovarianzterme zwischen den Anfalljahren hinzu.

    Returns
    -------
    DataFrame mit Index Anfalljahr plus Zeile "total" und den Spalten
    ultimate, reserve, se. Für das älteste Anfalljahr ist se = 0.
    """
    raise NotImplementedError

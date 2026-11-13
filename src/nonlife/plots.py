"""Grafiken für die README. Reine Darstellung, keine Berechnungen."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ohne Fenster, damit die CLI auch ohne Display läuft

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

SERIES_1 = "#2a78d6"  # Schätzung / Simulation
SERIES_2 = "#eb6834"  # Wahrheit / Theorie
SERIES_3 = "#1baf7a"  # einfache Auswertung (Vergleich)
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _new_figure() -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150, facecolor=SURFACE)
    _style(ax)
    return fig, ax


def _thousands(x: float, _pos: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


def save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_aggregate_distribution(
    simulations: np.ndarray,
    theory: Mapping[str, float],
    quantile_99: float,
) -> Figure:
    """Histogramm des simulierten Gesamtschadens mit E[S] und E[S] ± sd[S] laut Theorie."""
    fig, ax = _new_figure()
    ax.hist(simulations, bins=80, color=SERIES_1, edgecolor=SURFACE, linewidth=0.5,
            label="Monte Carlo")
    mean, std = theory["mean"], theory["std"]
    ax.axvline(mean, color=SERIES_2, linewidth=2, label="E[S] (Theorie)")
    ax.axvspan(mean - std, mean + std, color=SERIES_2, alpha=0.12, label="E[S] ± sd[S] (Theorie)")
    ax.axvline(quantile_99, color=TEXT_MUTED, linewidth=1.5, linestyle="--",
               label="99-%-Quantil (Monte Carlo)")
    ax.xaxis.set_major_formatter(_thousands)
    ax.set_xlabel("Jahresgesamtschaden S in EUR", color=TEXT)
    ax.set_ylabel("Anzahl Simulationsläufe", color=TEXT)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    return fig


def plot_reserves(
    estimated: pd.Series, true: pd.Series, se: pd.Series | None = None, z: float = 1.96
) -> Figure:
    """Geschätzte Reserve je Anfalljahr gegen die wahre, mit ±z·se, falls ``se`` gegeben."""
    fig, ax = _new_figure()
    x = np.arange(len(estimated))
    if se is None:
        ax.plot(x, estimated.to_numpy(), "o", color=SERIES_1, markersize=6, label="Chain-Ladder")
    else:
        ax.errorbar(x, estimated.to_numpy(), yerr=z * se.to_numpy(), fmt="o", color=SERIES_1,
                    markersize=6, capsize=3, linewidth=1.5,
                    label=f"Chain-Ladder ± {z:g}·se (Mack)")
    ax.plot(x, true.to_numpy(), "D", color=SERIES_2, markersize=6, label="wahre Reserve")
    ax.set_xticks(x, [str(i) for i in estimated.index])
    ax.yaxis.set_major_formatter(_thousands)
    ax.set_xlabel("Anfalljahr", color=TEXT)
    ax.set_ylabel("Reserve in EUR", color=TEXT)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    return fig


def plot_relativities(
    rel: pd.DataFrame,
    true: Mapping[tuple[str, str], float],
    one_way: Mapping[tuple[str, str], float] | None = None,
    confidence: float = 0.95,
) -> Figure:
    """Geschätzte Tariffaktoren mit Konfidenzintervall gegen die wahren Faktoren.

    ``rel`` im Format von :func:`nonlife.pricing.relativities`, Reihenfolge
    der Zeilen = Reihenfolge von oben nach unten. Mit ``one_way`` wird je
    Ausprägung zusätzlich die einfache Auswertung leicht versetzt gezeigt.
    """
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(rel) + 1.7), dpi=150, facecolor=SURFACE)
    _style(ax)
    y = np.arange(len(rel))[::-1].astype(float)
    if one_way is not None:
        ax.plot([one_way[k] for k in rel.index], y + 0.28, "s", color=SERIES_3, markersize=6,
                markeredgecolor=SURFACE, label="einfache Auswertung je Merkmal")
    err = np.vstack([rel["factor"] - rel["lower"], rel["upper"] - rel["factor"]])
    ax.errorbar(rel["factor"], y, xerr=err, fmt="o", color=SERIES_1, markersize=6, capsize=3,
                linewidth=1.5, label=f"GLM mit {confidence * 100:.0f}-%-Konfidenzintervall")
    ax.plot([true[k] for k in rel.index], y, "D", color=SERIES_2, markersize=6,
            markeredgecolor=SURFACE, label="wahrer Faktor")
    ax.axvline(1.0, color=TEXT_MUTED, linewidth=0.8)
    features = rel.index.get_level_values(0)
    for i in range(1, len(rel)):
        if features[i] != features[i - 1]:
            ax.axhline(y[i] + 0.5, color=GRID, linewidth=0.8)
    ax.set_yticks(y, [f"{f}: {lvl}" for f, lvl in rel.index])
    ax.set_ylim(-0.6, len(rel) - 0.4)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Faktor relativ zur Referenzklasse", color=TEXT)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.3)
    return fig


def plot_actual_vs_expected(
    tables: Mapping[str, pd.DataFrame], columns: Sequence[Sequence[str]], z: float = 1.96
) -> Figure:
    """Ist/Soll je Klasse, ein Feld je Merkmal, Merkmale in ``columns`` angeordnet.

    ``tables`` im Format von :func:`nonlife.evaluation.actual_vs_expected`.
    Die Balken reichen von Verhältnis − z/√Soll bis Verhältnis + z/√Soll:
    Unter dem Modell ist Ist ungefähr Poisson mit Mittel Soll, das Verhältnis
    streut also mit Standardabweichung 1/√Soll. Liegt die 1 außerhalb, ist
    die Abweichung mehr als Zufall (bei φ = 1; mit Überdispersion breiter).
    """
    rows = [sum(len(tables[f]) for f in col) for col in columns]
    fig = plt.figure(figsize=(9.5, 0.24 * max(rows) + 1.6), dpi=150, facecolor=SURFACE)
    outer = fig.add_gridspec(1, len(columns), wspace=0.45)
    lo = min(float((t["ratio"] - z / np.sqrt(t["expected"])).min()) for t in tables.values())
    hi = max(float((t["ratio"] + z / np.sqrt(t["expected"])).max()) for t in tables.values())
    for c, features in enumerate(columns):
        inner = outer[c].subgridspec(
            len(features), 1, height_ratios=[len(tables[f]) + 1.2 for f in features], hspace=0.3
        )
        for r, feature in enumerate(features):
            table = tables[feature]
            ax = fig.add_subplot(inner[r])
            _style(ax)
            y = np.arange(len(table))[::-1].astype(float)
            ax.errorbar(table["ratio"], y, xerr=z / np.sqrt(table["expected"]), fmt="o",
                        color=SERIES_1, markersize=4, capsize=2, linewidth=1.2)
            ax.axvline(1.0, color=TEXT_MUTED, linewidth=0.8)
            ax.set_yticks(y, [str(level) for level in table.index], fontsize=8)
            ax.set_ylim(-0.7, len(table) - 0.3)
            ax.set_xlim(lo - 0.05, hi + 0.05)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            ax.grid(axis="y", visible=False)
            ax.set_title(feature, loc="left", fontsize=9, color=TEXT, pad=3)
            if r < len(features) - 1:
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel(f"Ist / Soll (Testdaten), Balken ± {z:g}/√Soll", color=TEXT,
                              fontsize=9)
    return fig

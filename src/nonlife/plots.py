"""Grafiken für die README. Reine Darstellung, keine Berechnungen."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ohne Fenster, damit die CLI auch ohne Display läuft

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

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


def _feature_panels(
    sizes: Mapping[str, int],
    columns: Sequence[Sequence[str]],
    draw: Callable[[Axes, str, np.ndarray], None],
    xlabel: str,
) -> tuple[Figure, list[Axes]]:
    """Raster mit einem Feld je Merkmal, Merkmale spaltenweise wie in ``columns``.

    ``sizes`` ist die Anzahl der Klassen je Merkmal; die Feldhöhe richtet sich
    danach, damit jede Klasse gleich viel Platz bekommt. ``draw`` zeichnet ein
    Feld und erhält die y-Positionen der Klassen (oberste Klasse oben).
    """
    rows = [sum(sizes[f] for f in col) for col in columns]
    height = 0.24 * max(rows) + 1.8
    fig = plt.figure(figsize=(9.5, height), dpi=150, facecolor=SURFACE)
    # oben 0,5 Zoll für die Legende, unten 0,6 Zoll für die Achsenbeschriftung
    outer = fig.add_gridspec(1, len(columns), wspace=0.45, top=1 - 0.5 / height,
                             bottom=0.6 / height)
    axes: list[Axes] = []
    for c, features in enumerate(columns):
        inner = outer[c].subgridspec(
            len(features), 1, height_ratios=[sizes[f] + 1.2 for f in features], hspace=0.3
        )
        for r, feature in enumerate(features):
            ax = fig.add_subplot(inner[r])
            _style(ax)
            y = np.arange(sizes[feature])[::-1].astype(float)
            draw(ax, feature, y)
            ax.set_ylim(-0.7, sizes[feature] - 0.3)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            ax.grid(axis="y", visible=False)
            ax.set_title(feature, loc="left", fontsize=9, color=TEXT, pad=3)
            if r < len(features) - 1:
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel(xlabel, color=TEXT, fontsize=9)
            axes.append(ax)
    return fig, axes


def plot_actual_vs_expected(
    tables: Mapping[str, pd.DataFrame], columns: Sequence[Sequence[str]], z: float = 1.96
) -> Figure:
    """Ist/Soll je Klasse, ein Feld je Merkmal, Merkmale in ``columns`` angeordnet.

    ``tables`` im Format von :func:`nonlife.evaluation.actual_vs_expected`.
    Der Strich bei 1 heißt Ist = Soll. Die Balken reichen von
    Verhältnis − z/√Soll bis Verhältnis + z/√Soll: Unter dem Modell ist Ist
    ungefähr Poisson mit Mittel Soll, das Verhältnis streut also mit
    Standardabweichung 1/√Soll. Schneidet der Balken die 1 nicht, ist die
    Abweichung mehr als Zufall (bei φ = 1; mit Überdispersion breiter).
    """
    lo = min(float((t["ratio"] - z / np.sqrt(t["expected"])).min()) for t in tables.values())
    hi = max(float((t["ratio"] + z / np.sqrt(t["expected"])).max()) for t in tables.values())

    def draw(ax: Axes, feature: str, y: np.ndarray) -> None:
        table = tables[feature]
        ax.axvline(1.0, color=SERIES_2, linewidth=1.2)
        ax.errorbar(table["ratio"], y, xerr=z / np.sqrt(table["expected"]), fmt="o",
                    color=SERIES_1, markersize=4, capsize=2, linewidth=1.2)
        ax.set_yticks(y, [str(level) for level in table.index], fontsize=8)
        ax.set_xlim(lo - 0.05, hi + 0.05)

    fig, _ = _feature_panels(
        {f: len(t) for f, t in tables.items()}, columns, draw,
        f"Ist / Soll auf den Testdaten, Balken ± {z:g}/√Soll",
    )
    fig.legend(
        handles=[
            Line2D([], [], color=SERIES_1, marker="o", markersize=4,
                   label="Ist / Soll je Klasse mit Zufallsstreuung"),
            Line2D([], [], color=SERIES_2, linewidth=1.2, label="Ist = Soll (Vorhersage trifft)"),
        ],
        loc="upper left", bbox_to_anchor=(0.08, 1.0), ncol=2, frameon=False, fontsize=9,
        labelcolor=TEXT,
    )
    return fig


def plot_relativity_panels(
    rel: pd.DataFrame,
    columns: Sequence[Sequence[str]],
    one_way: Mapping[tuple[str, str], float] | None = None,
    confidence: float = 0.95,
    interval_note: str = "",
) -> Figure:
    """Geschätzte Tariffaktoren je Klasse, ein Feld je Merkmal, logarithmische Achse.

    ``rel`` im Format von :func:`nonlife.pricing.relativities`, Reihenfolge der
    Zeilen je Merkmal = Reihenfolge von oben nach unten. Die Referenzklasse hat
    den Faktor 1. Logarithmische Achse, weil Faktoren multiplikativ wirken:
    0,5 und 2 sind gleich weit von 1 entfernt. Mit ``one_way`` wird je Klasse
    zusätzlich die einfache Auswertung leicht versetzt gezeigt.
    """
    sizes = {f: len(rel.loc[f]) for f in dict.fromkeys(rel.index.get_level_values(0))}
    values = [rel["lower"], rel["upper"]]
    if one_way is not None:
        values.append(pd.Series(list(one_way.values())))
    lo = min(float(v.min()) for v in values) / 1.1
    hi = max(float(v.max()) for v in values) * 1.1

    def draw(ax: Axes, feature: str, y: np.ndarray) -> None:
        part = rel.loc[feature]
        ax.axvline(1.0, color=TEXT_MUTED, linewidth=0.8)
        if one_way is not None:
            ax.plot([one_way[(feature, str(lvl))] for lvl in part.index], y + 0.3, "s",
                    color=SERIES_3, markersize=4, markeredgecolor=SURFACE)
        err = np.vstack([part["factor"] - part["lower"], part["upper"] - part["factor"]])
        ax.errorbar(part["factor"], y, xerr=err, fmt="o", color=SERIES_1, markersize=4,
                    capsize=2, linewidth=1.2)
        ax.set_yticks(y, [str(level) for level in part.index], fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ticks = [t for t in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8) if lo <= t <= hi]
        ax.set_xticks(ticks, [f"{t:g}".replace(".", ",") for t in ticks])
        ax.xaxis.set_minor_locator(NullLocator())

    fig, _ = _feature_panels(sizes, columns, draw, "Faktor relativ zur Referenzklasse")
    handles = [Line2D([], [], color=SERIES_1, marker="o", markersize=4,
                      label=f"GLM mit {confidence * 100:.0f}-%-Konfidenzintervall{interval_note}")]
    if one_way is not None:
        handles.append(Line2D([], [], color=SERIES_3, marker="s", markersize=4, linestyle="",
                              label="einfache Auswertung je Merkmal"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.08, 1.0), ncol=2,
               frameon=False, fontsize=9, labelcolor=TEXT)
    return fig

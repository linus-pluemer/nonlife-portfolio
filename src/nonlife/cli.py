"""Kommandozeile: ``nonlife run`` erzeugt Daten, Kennzahlen und Grafiken neu.

Teile, deren Kernfunktionen noch ``NotImplementedError`` werfen, werden
übersprungen und als offen gemeldet.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from nonlife import collective, plots
from nonlife.data import save_portfolio
from nonlife.simulate import Portfolio, default_parameters, simulate_portfolio

VALUATION_YEAR = 2024


def _run_collective(
    portfolio: Portfolio, figures_dir: Path, n_sim: int, seed: int
) -> dict[str, Any]:
    policies, claims = portfolio.policies, portfolio.claims
    # S ist die Jahresschadenlast des Portfolios: Frequenz je Exposure-Jahr
    # mal mittlere Exposure eines Anfalljahres.
    frequency = collective.estimate_frequency_parameter(claims, policies)
    exposure_per_year = policies.groupby("accident_year")["exposure"].sum().mean()
    lambda_ = float(frequency * exposure_per_year)
    mu, sigma = collective.estimate_severity_parameters(claims)
    severity = stats.lognorm(s=sigma, scale=float(np.exp(mu)))

    simulations = collective.monte_carlo_simulation(
        lambda_, mu, sigma, n_sim, np.random.default_rng(seed)
    )
    sim = collective.calculate_statistics(simulations)
    theory = collective.theoretical_statistics(lambda_, severity.mean(), severity.var())

    print(f"  Frequenz je Exposure-Jahr = {frequency:.4f}, "
          f"Exposure je Jahr = {exposure_per_year:,.0f}")
    print(f"  λ = {lambda_:.1f}, μ = {mu:.4f}, σ = {sigma:.4f}")
    print(f"  {'':<10}{'Simulation':>20}{'Theorie':>20}")
    for key in ("mean", "variance", "std"):
        print(f"  {key:<10}{sim[key]:>20,.2f}{theory[key]:>20,.2f}")
    print(f"  99-%-Quantil (Simulation): {sim['quantile_99']:,.2f}")

    fig = plots.plot_aggregate_distribution(simulations, theory, sim["quantile_99"])
    path = plots.save(fig, figures_dir / "aggregate_loss.png")
    print(f"  Grafik: {path}")
    return {
        "frequency": frequency, "exposure_per_year": float(exposure_per_year),
        "lambda": lambda_, "mu": mu, "sigma": sigma,
        "simulation": {k: float(v) for k, v in sim.items()},
        "theory": {k: float(v) for k, v in theory.items()},
    }


def _run_reserving(portfolio: Portfolio, figures_dir: Path) -> dict[str, Any]:
    raise NotImplementedError("Dreieck noch nicht auf payments umgestellt")


def _run_pricing(portfolio: Portfolio, figures_dir: Path) -> dict[str, Any]:
    raise NotImplementedError


def _step(name: str, func: Callable[[], dict[str, Any]], results: dict[str, Any]) -> None:
    print(f"\n[{name}]")
    try:
        results[name] = func()
    except NotImplementedError as exc:
        results[name] = "offen"
        print(f"  offen{': ' + str(exc) if str(exc) else ''}")


def run(args: argparse.Namespace) -> None:
    data_dir: Path = args.data_dir
    figures_dir: Path = args.figures_dir

    years = range(VALUATION_YEAR - args.n_years + 1, VALUATION_YEAR + 1)
    portfolio = simulate_portfolio(
        default_parameters(), args.n_policies, years, np.random.default_rng(args.seed)
    )
    save_portfolio(portfolio, data_dir)
    print(
        f"{len(portfolio.policies)} Policen, {len(portfolio.claims)} Schäden, "
        f"{len(portfolio.payments)} Zahlungen, Anfalljahre {years[0]}-{years[-1]} -> {data_dir}"
    )

    results: dict[str, Any] = {
        "seed": args.seed, "n_policies_per_year": args.n_policies, "accident_years": list(years),
    }
    _step("collective",
          lambda: _run_collective(portfolio, figures_dir, args.n_sim, args.seed), results)
    _step("reserving", lambda: _run_reserving(portfolio, figures_dir), results)
    _step("pricing", lambda: _run_pricing(portfolio, figures_dir), results)

    out = figures_dir.parent / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nKennzahlen: {out}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nonlife", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Daten, Kennzahlen und Grafiken neu erzeugen")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--n-policies", type=int, default=10_000, help="Policen je Anfalljahr")
    p_run.add_argument("--n-years", type=int, default=10, help=f"Anfalljahre bis {VALUATION_YEAR}")
    p_run.add_argument("--n-sim", type=int, default=10_000, help="Monte-Carlo-Läufe")
    p_run.add_argument("--data-dir", type=Path, default=Path("data"))
    p_run.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    p_run.set_defaults(func=run)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows-Konsole: sonst cp1252, kein λ
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

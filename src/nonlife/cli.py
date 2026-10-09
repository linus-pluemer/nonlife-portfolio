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
from nonlife.data import load_portfolio, save_portfolio
from nonlife.simulate import simulate_claims, simulate_policies


def _run_collective(data_dir: Path, figures_dir: Path, n_sim: int, seed: int) -> dict[str, Any]:
    policies, claims = load_portfolio(data_dir)
    lambda_ = collective.estimate_frequency_parameter(claims, policies)
    mu, sigma = collective.estimate_severity_parameters(claims)
    severity = stats.lognorm(s=sigma, scale=float(np.exp(mu)))

    simulations = collective.monte_carlo_simulation(
        lambda_, mu, sigma, num_simulations=n_sim, random_seed=seed
    )
    sim = collective.calculate_statistics(simulations)
    theory = collective.theoretical_statistics(lambda_, severity.mean(), severity.var())

    print(f"  λ = {lambda_:.6f}, μ = {mu:.4f}, σ = {sigma:.4f}")
    print(f"  {'':<10}{'Simulation':>16}{'Theorie':>16}")
    for key in ("mean", "variance", "std"):
        print(f"  {key:<10}{sim[key]:>16,.2f}{theory[key]:>16,.2f}")
    print(f"  99-%-Quantil (Simulation): {sim['quantile_99']:,.2f}")

    fig = plots.plot_aggregate_distribution(simulations, theory, sim["quantile_99"])
    path = plots.save(fig, figures_dir / "aggregate_loss.png")
    print(f"  Grafik: {path}")
    return {
        "lambda": lambda_, "mu": mu, "sigma": sigma,
        "simulation": {k: float(v) for k, v in sim.items()},
        "theory": {k: float(v) for k, v in theory.items()},
    }


def _run_reserving(data_dir: Path, figures_dir: Path) -> dict[str, Any]:
    _, claims = load_portfolio(data_dir)
    if not {"accident_year", "development_year"} <= set(claims.columns):
        raise NotImplementedError("Simulation liefert noch kein Anfall- und Abwicklungsjahr")
    raise NotImplementedError


def _run_pricing(data_dir: Path, figures_dir: Path) -> dict[str, Any]:
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

    policies = simulate_policies(n_policies=args.n_policies, seed=args.seed)
    claims = simulate_claims(policies, seed=args.seed)
    save_portfolio(policies, claims, data_dir)
    print(f"{len(policies)} Policen, {len(claims)} Schäden -> {data_dir}")

    results: dict[str, Any] = {"seed": args.seed, "n_policies": args.n_policies}
    _step("collective",
          lambda: _run_collective(data_dir, figures_dir, args.n_sim, args.seed), results)
    _step("reserving", lambda: _run_reserving(data_dir, figures_dir), results)
    _step("pricing", lambda: _run_pricing(data_dir, figures_dir), results)

    out = figures_dir.parent / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nKennzahlen: {out}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nonlife", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Daten, Kennzahlen und Grafiken neu erzeugen")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--n-policies", type=int, default=5000)
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

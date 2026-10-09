"""Tests für nonlife.data (aus den Selbsttests von Phase 1 hervorgegangen)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nonlife.data import (
    DataValidationError,
    load_portfolio,
    observed_frequency_by,
    save_portfolio,
    validate_claims,
    validate_policies,
)
from nonlife.simulate import simulate_claims, simulate_policies


@pytest.fixture
def portfolio() -> tuple[pd.DataFrame, pd.DataFrame]:
    policies = simulate_policies(n_policies=500, seed=3)
    return policies, simulate_claims(policies, seed=3)


def test_roundtrip(tmp_path: Path, portfolio: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    policies, claims = portfolio
    save_portfolio(policies, claims, tmp_path)
    loaded_policies, loaded_claims = load_portfolio(tmp_path)
    pd.testing.assert_frame_equal(loaded_policies, policies)
    pd.testing.assert_frame_equal(loaded_claims, claims, check_dtype=False)


def test_simulated_portfolio_is_valid(portfolio: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    policies, claims = portfolio
    validate_policies(policies)
    validate_claims(claims, policies)


def test_missing_column_is_rejected(portfolio: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    policies, _ = portfolio
    with pytest.raises(DataValidationError, match="exposure"):
        validate_policies(policies.drop(columns="exposure"))


def test_exposure_out_of_range_is_rejected(portfolio: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    policies, _ = portfolio
    broken = policies.copy()
    broken.loc[0, "exposure"] = 1.5
    with pytest.raises(DataValidationError, match="exposure"):
        validate_policies(broken)


def test_orphan_claim_is_rejected(portfolio: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    policies, claims = portfolio
    broken = claims.copy()
    broken.loc[broken.index[0], "policy_id"] = -1
    with pytest.raises(DataValidationError, match="policy_id"):
        validate_claims(broken, policies)


def test_non_positive_amount_is_rejected(portfolio: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    policies, claims = portfolio
    broken = claims.copy()
    broken.loc[broken.index[0], "claim_amount"] = 0.0
    with pytest.raises(DataValidationError, match="claim_amount"):
        validate_claims(broken, policies)


def test_observed_frequency_by_hand() -> None:
    policies = pd.DataFrame(
        {"policy_id": [1, 2, 3], "exposure": [1.0, 0.5, 0.5], "region": ["A", "A", "B"]}
    )
    claims = pd.DataFrame({"policy_id": [1, 1, 2], "claim_amount": [1.0, 1.0, 1.0]})
    result = observed_frequency_by(policies, claims, "region").set_index("region")
    assert result.loc["A", "n_claims"] == 3
    assert result.loc["A", "frequency"] == pytest.approx(2.0)
    assert result.loc["B", "n_claims"] == 0
    assert result.loc["B", "frequency"] == 0.0


def test_observed_frequency_totals(portfolio: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    policies, claims = portfolio
    result = observed_frequency_by(policies, claims, "region")
    assert result["n_claims"].sum() == len(claims)
    assert result["exposure"].sum() == pytest.approx(policies["exposure"].sum())

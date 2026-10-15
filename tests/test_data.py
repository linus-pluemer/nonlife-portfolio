"""Tests für nonlife.data (aus den Selbsttests von Phase 1 hervorgegangen)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nonlife.data import (
    DataValidationError,
    load_portfolio,
    observed_frequency_by,
    save_portfolio,
    validate_claims,
    validate_payments,
    validate_policies,
)
from nonlife.simulate import Portfolio, default_parameters, simulate_portfolio


@pytest.fixture(scope="module")
def portfolio() -> Portfolio:
    return simulate_portfolio(
        default_parameters(), 500, range(2020, 2025), np.random.default_rng(3)
    )


def test_roundtrip(tmp_path: Path, portfolio: Portfolio) -> None:
    save_portfolio(portfolio, tmp_path)
    loaded = load_portfolio(tmp_path)
    pd.testing.assert_frame_equal(loaded.policies, portfolio.policies, check_dtype=False)
    pd.testing.assert_frame_equal(loaded.claims, portfolio.claims, check_dtype=False)
    pd.testing.assert_frame_equal(loaded.payments, portfolio.payments, check_dtype=False)


def test_simulated_portfolio_is_valid(portfolio: Portfolio) -> None:
    validate_policies(portfolio.policies)
    validate_claims(portfolio.claims, portfolio.policies)
    validate_payments(portfolio.payments, portfolio.claims)


def test_missing_column_is_rejected(portfolio: Portfolio) -> None:
    with pytest.raises(DataValidationError, match="exposure"):
        validate_policies(portfolio.policies.drop(columns="exposure"))


def test_exposure_out_of_range_is_rejected(portfolio: Portfolio) -> None:
    broken = portfolio.policies.copy()
    broken.loc[0, "exposure"] = 1.5
    with pytest.raises(DataValidationError, match="exposure"):
        validate_policies(broken)


def test_orphan_claim_is_rejected(portfolio: Portfolio) -> None:
    broken = portfolio.claims.copy()
    broken.loc[broken.index[0], "policy_id"] = -1
    with pytest.raises(DataValidationError, match="policy_id"):
        validate_claims(broken, portfolio.policies)


def test_claim_year_must_match_policy(portfolio: Portfolio) -> None:
    broken = portfolio.claims.copy()
    broken.loc[broken.index[0], "accident_year"] += 1
    with pytest.raises(DataValidationError, match="accident_year"):
        validate_claims(broken, portfolio.policies)


def test_non_positive_amount_is_rejected(portfolio: Portfolio) -> None:
    broken = portfolio.claims.copy()
    broken.loc[broken.index[0], "claim_amount"] = 0.0
    with pytest.raises(DataValidationError, match="claim_amount"):
        validate_claims(broken, portfolio.policies)


def test_overpayment_is_rejected(portfolio: Portfolio) -> None:
    broken = portfolio.payments.copy()
    broken.loc[broken.index[0], "amount"] *= 1e6
    with pytest.raises(DataValidationError, match="übersteigen"):
        validate_payments(broken, portfolio.claims)


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


def test_observed_frequency_totals(portfolio: Portfolio) -> None:
    policies, claims = portfolio.policies, portfolio.claims
    result = observed_frequency_by(policies, claims, "region")
    assert result["n_claims"].sum() == len(claims)
    assert result["exposure"].sum() == pytest.approx(policies["exposure"].sum())

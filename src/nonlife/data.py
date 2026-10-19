"""Laden, Validieren und Aggregieren der Portfoliodaten."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd

from nonlife.simulate import Portfolio

DEFAULT_DATA_DIR = Path("data")

POLICY_COLUMNS: tuple[str, ...] = ("policy_id", "accident_year", "exposure")
CLAIM_COLUMNS: tuple[str, ...] = ("claim_id", "policy_id", "accident_year", "claim_amount")
PAYMENT_COLUMNS: tuple[str, ...] = ("claim_id", "accident_year", "development_year", "amount")

class DataValidationError(ValueError):
    """Die Daten verletzen eine erwartete Struktur oder einen Wertebereich."""


class PortfolioTables(NamedTuple):
    """Die drei Tabellen eines Portfolios, wie sie auf der Platte liegen."""

    policies: pd.DataFrame
    claims: pd.DataFrame
    payments: pd.DataFrame


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"{name}: fehlende Spalten {missing}")


def validate_policies(policies: pd.DataFrame) -> None:
    """Prüft Spalten, Eindeutigkeit der policy_id und Wertebereich der Exposure.

    Tarifmerkmale sind frei wählbar und werden nur auf fehlende Werte geprüft.

    Raises
    ------
    DataValidationError
        bei der ersten verletzten Bedingung.
    """
    _require_columns(policies, POLICY_COLUMNS, "policies")
    if not policies["policy_id"].is_unique:
        raise DataValidationError("policies: policy_id ist nicht eindeutig")
    if not policies["exposure"].between(0, 1, inclusive="right").all():
        raise DataValidationError("policies: exposure muss in (0, 1] liegen")
    if policies.isna().any().any():
        raise DataValidationError("policies: fehlende Werte")


def validate_claims(claims: pd.DataFrame, policies: pd.DataFrame) -> None:
    """Prüft Spalten, Eindeutigkeit der claim_id, positive Beträge und Fremdschlüssel.

    Das Anfalljahr eines Schadens muss mit dem seiner Police übereinstimmen.

    Raises
    ------
    DataValidationError
        bei der ersten verletzten Bedingung.
    """
    _require_columns(claims, CLAIM_COLUMNS, "claims")
    if not claims["claim_id"].is_unique:
        raise DataValidationError("claims: claim_id ist nicht eindeutig")
    if not (claims["claim_amount"] > 0).all():
        raise DataValidationError("claims: claim_amount muss positiv sein")
    year = claims["policy_id"].map(policies.set_index("policy_id")["accident_year"])
    if year.isna().any():
        raise DataValidationError("claims: policy_id ohne zugehörige Police")
    if not (year == claims["accident_year"]).all():
        raise DataValidationError("claims: accident_year weicht von der Police ab")


def validate_payments(payments: pd.DataFrame, claims: pd.DataFrame) -> None:
    """Prüft Spalten, Fremdschlüssel, Wertebereiche und dass nie mehr als der Schaden gezahlt wird.

    Raises
    ------
    DataValidationError
        bei der ersten verletzten Bedingung.
    """
    _require_columns(payments, PAYMENT_COLUMNS, "payments")
    amount = claims.set_index("claim_id")["claim_amount"]
    if not payments["claim_id"].isin(amount.index).all():
        raise DataValidationError("payments: claim_id ohne zugehörigen Schaden")
    if not (payments["development_year"] >= 0).all():
        raise DataValidationError("payments: development_year muss nichtnegativ sein")
    if not (payments["amount"] >= 0).all():
        raise DataValidationError("payments: amount muss nichtnegativ sein")
    paid = payments.groupby("claim_id")["amount"].sum()
    if (paid > amount.loc[paid.index] * (1 + 1e-9)).any():
        raise DataValidationError("payments: Zahlungen übersteigen den Schaden")


def save_portfolio(portfolio: Portfolio, data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Schreibt policies.csv, claims.csv und payments.csv nach ``data_dir``.

    Die wahren Parameter werden nicht gespeichert; sie stehen in ``portfolio.params``.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    portfolio.policies.to_csv(data_dir / "policies.csv", index=False)
    portfolio.claims.to_csv(data_dir / "claims.csv", index=False)
    portfolio.payments.to_csv(data_dir / "payments.csv", index=False)


def load_portfolio(data_dir: Path = DEFAULT_DATA_DIR) -> PortfolioTables:
    """Lädt und validiert policies.csv, claims.csv und payments.csv aus ``data_dir``."""
    tables = PortfolioTables(
        policies=pd.read_csv(data_dir / "policies.csv"),
        claims=pd.read_csv(data_dir / "claims.csv"),
        payments=pd.read_csv(data_dir / "payments.csv"),
    )
    validate_policies(tables.policies)
    validate_claims(tables.claims, tables.policies)
    validate_payments(tables.payments, tables.claims)
    return tables


def observed_frequency_by(
    policies: pd.DataFrame, claims: pd.DataFrame, feature: str
) -> pd.DataFrame:
    """Beobachtete Schadenfrequenz (Schäden je Exposure-Jahr) je Ausprägung von ``feature``.

    Returns
    -------
    DataFrame mit den Spalten ``feature``, n_claims, exposure, frequency.
    Ausprägungen ohne Schaden erscheinen mit n_claims = 0.
    """
    merged = claims.merge(policies[["policy_id", feature]], on="policy_id", how="left")
    n_claims = merged.groupby(feature).size().rename("n_claims")
    exposure = policies.groupby(feature)["exposure"].sum().rename("exposure")

    result = pd.concat([n_claims, exposure], axis=1).fillna({"n_claims": 0})
    result["n_claims"] = result["n_claims"].astype(int)
    result["frequency"] = result["n_claims"] / result["exposure"]
    return result.rename_axis(feature).reset_index()

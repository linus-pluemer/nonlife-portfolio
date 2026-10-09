"""Laden, Validieren und Aggregieren der Portfoliodaten, Abruf von freMTPL2."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path("data")

POLICY_COLUMNS: tuple[str, ...] = (
    "policy_id",
    "exposure",
    "fahrer_alter",
    "ps_klasse",
    "region",
    "erwartete_frequenz",
)
CLAIM_COLUMNS: tuple[str, ...] = ("claim_id", "policy_id", "claim_amount")

FREMTPL2_FREQ_ID = 41214
FREMTPL2_SEV_ID = 41215


class DataValidationError(ValueError):
    """Die Daten verletzen eine erwartete Struktur oder einen Wertebereich."""


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"{name}: fehlende Spalten {missing}")


def validate_policies(policies: pd.DataFrame) -> None:
    """Prüft Spalten, Eindeutigkeit der policy_id und Wertebereich der Exposure.

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
    if policies[list(POLICY_COLUMNS)].isna().any().any():
        raise DataValidationError("policies: fehlende Werte")


def validate_claims(claims: pd.DataFrame, policies: pd.DataFrame) -> None:
    """Prüft Spalten, Eindeutigkeit der claim_id, positive Beträge und Fremdschlüssel.

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
    if not claims["policy_id"].isin(policies["policy_id"]).all():
        raise DataValidationError("claims: policy_id ohne zugehörige Police")


def save_portfolio(
    policies: pd.DataFrame, claims: pd.DataFrame, data_dir: Path = DEFAULT_DATA_DIR
) -> None:
    """Schreibt policies.csv und claims.csv nach ``data_dir``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    policies.to_csv(data_dir / "policies.csv", index=False)
    claims.to_csv(data_dir / "claims.csv", index=False)


def load_portfolio(data_dir: Path = DEFAULT_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lädt und validiert policies.csv und claims.csv aus ``data_dir``.

    Returns
    -------
    (policies, claims)
    """
    policies = pd.read_csv(data_dir / "policies.csv")
    claims = pd.read_csv(data_dir / "claims.csv")
    validate_policies(policies)
    validate_claims(claims, policies)
    return policies, claims


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


def fetch_fremtpl2(
    cache_dir: Path = DEFAULT_DATA_DIR / "openml",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lädt freMTPL2freq und freMTPL2sev von OpenML und speichert sie lokal zwischen.

    Beim ersten Aufruf ist eine Internetverbindung nötig, danach wird der Cache
    in ``cache_dir`` verwendet.

    Returns
    -------
    (freq, sev) -- freq mit einer Zeile je Police (IDpol, ClaimNb, Exposure, Merkmale),
    sev mit einer Zeile je Schaden (IDpol, ClaimAmount).
    """
    from sklearn.datasets import fetch_openml  # Import hier, damit sklearn nur bei Bedarf lädt

    freq = fetch_openml(data_id=FREMTPL2_FREQ_ID, as_frame=True, data_home=cache_dir).frame
    sev = fetch_openml(data_id=FREMTPL2_SEV_ID, as_frame=True, data_home=cache_dir).frame
    freq["IDpol"] = freq["IDpol"].astype(int)
    sev["IDpol"] = sev["IDpol"].astype(int)
    return freq, sev

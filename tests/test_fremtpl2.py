"""Tests für nonlife.fremtpl2.

Alle Tests außer dem letzten laufen gegen kleine künstliche Tabellen im
Format von OpenML, ohne Netz. Jede Police steht für einen Fall der
Bereinigung, sodass sich jeder Zähler im Bericht von Hand nachrechnen lässt.

Der letzte Test lädt die echten Daten und prüft die Kontrollwerte aus
Wüthrich und Merz (2023). Er läuft nur mit ``pytest -m network``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nonlife.data import DataValidationError
from nonlife.fremtpl2 import (
    BONUS_MALUS_LADDER,
    clean_fremtpl2,
    duplicate_groups,
    fetch_fremtpl2,
    tariff_features,
)


def _freq(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults: dict[str, object] = {
        "ClaimNb": 0, "Exposure": 1.0, "Area": "C", "VehPower": 6, "VehAge": 5,
        "DrivAge": 45, "BonusMalus": 50, "VehBrand": "B12", "VehGas": "'Regular'",
        "Density": 100, "Region": "R24",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


@pytest.fixture
def raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sechs Policen, je ein Fall; Kommentar = erwartetes Ergebnis."""
    freq = _freq(
        [
            {"IDpol": 1.0, "ClaimNb": 2},        # 1 Schadenhöhe -> n = 1, IDpol ≤ 24500
            {"IDpol": 2.0, "ClaimNb": 1},        # keine Schadenhöhe -> n = 0, IDpol ≤ 24500
            {"IDpol": 30000.0, "ClaimNb": 1},    # passt, unverändert
            {"IDpol": 30001.0, "ClaimNb": 0},    # 6 Schadenhöhen -> entfernt
            {"IDpol": 30002.0, "ClaimNb": 5},    # 5 Schadenhöhen -> bleibt
            {"IDpol": 30003.0, "Exposure": 1.3},  # Exposure -> 1
        ]
    )
    sev_ids = [1, 30000, *[30001] * 6, *[30002] * 5, 99999]  # 99999: Schaden ohne Police
    sev = pd.DataFrame({"IDpol": sev_ids, "ClaimAmount": 1000.0})
    return freq, sev


# --- Bereinigung -------------------------------------------------------------


def test_claim_counts_recomputed_from_severities(raw: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    policies, _, _ = clean_fremtpl2(*raw)
    n = policies.set_index("policy_id")["n_claims"]
    assert n.to_dict() == {1: 1, 2: 0, 30000: 1, 30002: 5, 30003: 0}


def test_cleaning_report_counts_each_step(raw: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    _, _, report = clean_fremtpl2(*raw)
    assert report.n_policies_raw == 6
    assert report.n_claims_raw == 9  # Σ ClaimNb = 2 + 1 + 1 + 0 + 5 + 0
    assert report.n_severity_rows_raw == 14
    assert report.n_claims_without_policy == 1
    assert report.n_claim_count_changed == 3  # IDpol 1, 2 und 30001
    assert report.n_claim_count_changed_low_id == 2
    assert report.n_policies_removed == 1
    assert report.n_claims_removed == 6
    assert report.n_exposure_capped == 1
    assert report.n_policies == 5
    assert report.n_claims == 7  # 1 + 1 + 5


def test_claims_match_claim_counts(raw: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    policies, claims, _ = clean_fremtpl2(*raw)
    assert claims["policy_id"].isin(policies["policy_id"]).all()
    assert claims["claim_id"].is_unique
    assert len(claims) == policies["n_claims"].sum()


def test_exposure_capped_at_one(raw: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    policies, _, _ = clean_fremtpl2(*raw)
    assert policies["exposure"].max() == 1.0


def test_text_features_without_quotes(raw: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    policies, _, _ = clean_fremtpl2(*raw)
    assert set(policies["veh_gas"]) == {"Regular"}


def test_nonpositive_exposure_rejected(raw: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    freq, sev = raw
    freq.loc[0, "Exposure"] = 0.0
    with pytest.raises(DataValidationError, match="Exposure"):
        clean_fremtpl2(freq, sev)


def test_missing_column_rejected(raw: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    freq, sev = raw
    with pytest.raises(DataValidationError, match="Region"):
        clean_fremtpl2(freq.drop(columns="Region"), sev)


@pytest.mark.parametrize(("table", "column"), [("freq", "Region"), ("sev", "ClaimAmount")])
def test_missing_values_rejected(
    raw: tuple[pd.DataFrame, pd.DataFrame], table: str, column: str
) -> None:
    freq, sev = raw
    frame = freq if table == "freq" else sev
    frame.loc[0, column] = None  # Text: sonst still "nan"; Zahl: sonst NaN im Schadenbetrag
    with pytest.raises(DataValidationError, match=f"{table}: fehlende Werte in \\['{column}'\\]"):
        clean_fremtpl2(freq, sev)


# --- Tarifmerkmale -----------------------------------------------------------


def test_bonus_malus_ladder() -> None:
    assert BONUS_MALUS_LADDER == (100, 95, 90, 85, 80, 76, 72, 68, 64, 60, 57, 54, 51, 50)


def _features(**raw_values: list[object]) -> pd.DataFrame:
    n = len(next(iter(raw_values.values())))
    base = {"area": "C", "veh_power": 6, "veh_age": 5, "driv_age": 45, "bonus_malus": 50,
            "veh_brand": "B12", "veh_gas": "Regular", "density": 100, "region": "R24"}
    return tariff_features(pd.DataFrame({k: raw_values.get(k, [v] * n) for k, v in base.items()}))


@pytest.mark.parametrize(
    ("feature", "values", "classes"),
    [
        ("veh_power", [4, 8, 9, 15], ["4", "8", "9+", "9+"]),
        ("veh_age", [0, 1, 10, 11], ["0", "1-10", "1-10", "11+"]),
        ("driv_age", [18, 20, 21, 70, 71], ["18-20", "18-20", "21-25", "51-70", "71+"]),
        ("bonus_malus", [50, 51, 57, 58, 72, 73, 100, 101, 230],
         ["50", "51-57", "51-57", "58-72", "58-72", "73-90", "91-100", "101+", "101+"]),
    ],
)
def test_class_boundaries(feature: str, values: list[object], classes: list[str]) -> None:
    assert _features(**{feature: values})[feature].astype(str).tolist() == classes


@pytest.mark.parametrize(
    ("feature", "value"), [("driv_age", 17), ("veh_power", 3), ("bonus_malus", 49)]
)
def test_value_below_lowest_class_rejected(feature: str, value: int) -> None:
    # pd.cut gäbe still NaN zurück, die Police hätte keine Klasse
    with pytest.raises(DataValidationError, match=f"{feature}: 1 Werte außerhalb"):
        _features(**{feature: [value]})


def test_off_ladder_marks_malus_values() -> None:
    # 62 = 50 · 1,25 (Schaden vom Boden aus), 101 und mehr nur nach Schaden
    result = _features(bonus_malus=[50, 62, 64, 100, 101])["bm_off_ladder"]
    assert result.tolist() == [False, True, False, False, True]


def test_density_not_a_tariff_feature() -> None:
    assert "density" not in _features(veh_age=[1]).columns


def test_duplicate_groups_on_raw_values() -> None:
    policies = pd.DataFrame(
        {"area": "C", "veh_power": 6, "veh_age": [5, 5, 6], "driv_age": 45, "bonus_malus": 50,
         "veh_brand": "B12", "veh_gas": "Regular", "density": 100, "region": "R24"}
    )
    groups = duplicate_groups(policies)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]  # Fahrzeugalter 6 statt 5, gleiche Klasse 1-10


# --- echte Daten -------------------------------------------------------------


@pytest.mark.network
def test_control_values_wuethrich_merz() -> None:
    """Kontrollwerte aus Wüthrich und Merz (2023), Anhang B, und eigener Zählung.

    Die Frequenz wird mit Toleranz geprüft: Das Buch nennt 7,35 %, seine
    Tabelle 13.1 ergibt aber 26 383 / 358 359 = 7,36 %.
    """
    pytest.importorskip("sklearn")
    policies, claims, report = clean_fremtpl2(*fetch_fremtpl2(Path("data/openml")))
    assert report.n_policies == 678_007
    assert report.n_claims == 26_383
    assert report.n_claim_count_changed == 9_117
    assert report.n_claim_count_changed_low_id == 8_851
    frequency = policies["n_claims"].sum() / policies["exposure"].sum()
    assert frequency == pytest.approx(0.0736, abs=1e-4)
    assert len(claims) == policies["n_claims"].sum()

"""freMTPL2: französisches Kfz-Haftpflichtportfolio, Laden, Bereinigen, Tarifmerkmale.

Quelle: OpenML 41214 (``freMTPL2freq``, eine Zeile je Police) und 41215
(``freMTPL2sev``, eine Zeile je Schaden), entspricht CASdatasets 1.0-8.
Ein Beobachtungsjahr, rund 678 000 Policen.

Die Bereinigung folgt Wüthrich und Merz (2023), Statistical Foundations of
Actuarial Learning and its Applications, Anhang B, Listing 13.1:

1. ``ClaimNb`` ist für viele Policen mit IDpol ≤ 24500 fehlerhaft (es gibt
   keine Schadenhöhen dazu). Die Schadenzahl wird deshalb aus der Anzahl der
   Zeilen in ``freMTPL2sev`` neu berechnet. Keine Police wird dafür entfernt,
   sonst würden nur Policen mit Schaden wegfallen und die Frequenz sänke.
2. Policen mit mehr als 5 Schäden werden entfernt (vermutlich Datenfehler,
   sehr kurze Exposure).
3. Die Exposure wird bei 1 gekappt (ein Beobachtungsjahr).
4. Schäden ohne Police werden entfernt.

Kontrollwerte nach der Bereinigung: 678 007 Policen, 26 383 Schäden.

Die Klassen für Fahrzeugstärke, Fahrzeug- und Fahreralter stammen aus Noll,
Salzmann, Wüthrich (2018), Case Study: French Motor Third-Party Liability
Claims, SSRN 3164764, Abschnitt 3.1. ``Density`` geht nicht in die Klassen
ein: ``Area`` ist im Datensatz bereits eine Einteilung der Dichte (A: 1–50
bis F: über 10 000 Einwohner/km²), beide zusammen wären fast kollinear.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nonlife.data import DEFAULT_DATA_DIR, DataValidationError

FREMTPL2_FREQ_ID = 41214
FREMTPL2_SEV_ID = 41215

MAX_CLAIMS = 5
LOW_ID_LIMIT = 24_500

RAW_FEATURES: tuple[str, ...] = (
    "area", "veh_power", "veh_age", "driv_age", "bonus_malus",
    "veh_brand", "veh_gas", "density", "region",
)
TARIFF_FEATURES: tuple[str, ...] = (
    "area", "veh_power", "veh_age", "driv_age", "bonus_malus", "bm_off_ladder",
    "veh_brand", "veh_gas", "region",
)
_RENAME = {
    "IDpol": "policy_id", "ClaimNb": "claim_nb_raw", "Exposure": "exposure",
    "Area": "area", "VehPower": "veh_power", "VehAge": "veh_age", "DrivAge": "driv_age",
    "BonusMalus": "bonus_malus", "VehBrand": "veh_brand", "VehGas": "veh_gas",
    "Density": "density", "Region": "region",
}


def _bonus_malus_ladder(start: int = 100, floor: int = 50) -> tuple[int, ...]:
    """Werte des französischen Bonus-Malus bei lauter schadenfreien Jahren ab ``start``.

    Je schadenfreiem Jahr mal 0,95, abgerundet auf zwei Stellen (hier: ganze
    Zahlen, Koeffizient mal 100), bis zum Boden 0,50. Ganzzahlig gerechnet,
    damit keine Rundungsfehler der Gleitkommazahlen hineinspielen.
    """
    values = [start]
    while values[-1] > floor:
        values.append(max(floor, values[-1] * 95 // 100))
    return tuple(values)


BONUS_MALUS_LADDER: tuple[int, ...] = _bonus_malus_ladder()


@dataclass(frozen=True)
class CleaningReport:
    """Zählung jedes Bereinigungsschritts, damit die Bereinigung nachvollziehbar ist."""

    n_policies_raw: int
    n_claims_raw: int  # Σ ClaimNb vor der Bereinigung
    n_severity_rows_raw: int
    n_claims_without_policy: int
    n_claim_count_changed: int
    n_claim_count_changed_low_id: int  # davon IDpol ≤ 24500
    n_policies_removed: int  # mehr als MAX_CLAIMS Schäden
    n_claims_removed: int  # Schäden dieser Policen
    n_exposure_capped: int
    n_policies: int
    n_claims: int


def fetch_fremtpl2(
    cache_dir: Path = DEFAULT_DATA_DIR / "openml",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lädt freMTPL2freq und freMTPL2sev roh von OpenML und speichert sie lokal zwischen.

    Beim ersten Aufruf ist eine Internetverbindung nötig, danach wird der
    Cache in ``cache_dir`` verwendet. Braucht das optionale Extra
    ``fremtpl`` (scikit-learn).

    Returns
    -------
    (freq, sev) in den Originalspalten, unbereinigt; weiter mit
    :func:`clean_fremtpl2`.
    """
    from sklearn.datasets import fetch_openml  # erst hier, damit sklearn optional bleibt

    freq = fetch_openml(data_id=FREMTPL2_FREQ_ID, as_frame=True, data_home=cache_dir,
                        parser="auto").frame
    sev = fetch_openml(data_id=FREMTPL2_SEV_ID, as_frame=True, data_home=cache_dir,
                       parser="auto").frame
    return freq, sev


def clean_fremtpl2(
    freq: pd.DataFrame, sev: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Bereinigt freMTPL2 nach Wüthrich und Merz (2023), siehe Moduldocstring.

    Parameters
    ----------
    freq, sev
        Rohdaten in den Originalspalten, wie von :func:`fetch_fremtpl2`

    Returns
    -------
    policies
        eine Zeile je Police: policy_id, n_claims, exposure und die Rohmerkmale
        aus ``RAW_FEATURES`` (Text als ``str``, ohne Anführungszeichen)
    claims
        eine Zeile je Schaden: claim_id, policy_id, claim_amount
    report
        Zählung je Schritt

    Raises
    ------
    DataValidationError
        bei fehlenden Spalten, doppelter IDpol oder Exposure ≤ 0.
    """
    required = {"freq": (freq, list(_RENAME)), "sev": (sev, ["IDpol", "ClaimAmount"])}
    for name, (frame, columns) in required.items():
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise DataValidationError(f"freMTPL2 {name}: fehlende Spalten {missing}")

    policies = freq[list(_RENAME)].rename(columns=_RENAME)
    policies["policy_id"] = policies["policy_id"].astype(np.int64)
    if not policies["policy_id"].is_unique:
        raise DataValidationError("freMTPL2: IDpol ist nicht eindeutig")
    if not (policies["exposure"] > 0).all():
        raise DataValidationError("freMTPL2: Exposure muss positiv sein")
    for col in ("area", "veh_brand", "veh_gas", "region"):
        # OpenML liefert VehGas als "'Regular'"; Kategorien als einfache Zeichenketten.
        policies[col] = policies[col].astype(str).str.strip("'")

    claims = pd.DataFrame(
        {
            "policy_id": sev["IDpol"].astype(np.int64).to_numpy(),
            "claim_amount": sev["ClaimAmount"].astype(float).to_numpy(),
        }
    )
    n_claims_raw = int(policies["claim_nb_raw"].sum())

    # 4. Schäden ohne Police (zuerst, damit sie nicht mitgezählt werden)
    has_policy = claims["policy_id"].isin(policies["policy_id"])
    n_without_policy = int((~has_policy).sum())
    claims = claims[has_policy]

    # 1. Schadenzahl aus den Schadenhöhen neu berechnen
    counts = claims["policy_id"].value_counts()
    policies["n_claims"] = policies["policy_id"].map(counts).fillna(0).astype(np.int64)
    changed = policies["n_claims"] != policies["claim_nb_raw"]
    low_id = policies["policy_id"] <= LOW_ID_LIMIT

    # 2. Policen mit mehr als MAX_CLAIMS Schäden samt ihren Schäden entfernen
    too_many = policies["n_claims"] > MAX_CLAIMS
    removed_ids = policies.loc[too_many, "policy_id"]
    n_claims_removed = int(claims["policy_id"].isin(removed_ids).sum())
    policies = policies[~too_many]
    claims = claims[~claims["policy_id"].isin(removed_ids)]

    # 3. Exposure bei 1 kappen
    n_capped = int((policies["exposure"] > 1).sum())
    policies["exposure"] = policies["exposure"].clip(upper=1.0)

    policies = policies[["policy_id", "n_claims", "exposure", *RAW_FEATURES]].reset_index(drop=True)
    claims = claims.reset_index(drop=True)
    claims.insert(0, "claim_id", np.arange(len(claims)))

    report = CleaningReport(
        n_policies_raw=len(freq),
        n_claims_raw=n_claims_raw,
        n_severity_rows_raw=len(sev),
        n_claims_without_policy=n_without_policy,
        n_claim_count_changed=int(changed.sum()),
        n_claim_count_changed_low_id=int((changed & low_id).sum()),
        n_policies_removed=int(too_many.sum()),
        n_claims_removed=n_claims_removed,
        n_exposure_capped=n_capped,
        n_policies=len(policies),
        n_claims=len(claims),
    )
    return policies, claims, report


def _cut(values: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    """Intervalle [a, b) mit lesbaren Namen, als geordnete Kategorie."""
    return pd.cut(values, edges, right=False, labels=labels)


def _numbered(values: pd.Series) -> pd.Series:
    """Codes wie B1, B10, B2 oder R11, R24 als Kategorie in Zahlreihenfolge (B1, B2, …, B10)."""
    order = sorted(values.unique(), key=lambda code: int(code[1:]))
    return pd.Series(pd.Categorical(values, categories=order, ordered=True), index=values.index)


def tariff_features(policies: pd.DataFrame) -> pd.DataFrame:
    """Klassenmerkmale für das GLM aus den Rohmerkmalen von :func:`clean_fremtpl2`.

    * ``veh_power``: 4, 5, 6, 7, 8, 9+ (Paper: Stärke ab 9 zusammengefasst)
    * ``veh_age``: 0, 1–10, 11+ Jahre (Paper: [0, 1), [1, 10], (10, ∞))
    * ``driv_age``: 18–20, 21–25, 26–30, 31–40, 41–50, 51–70, 71+ (Paper)
    * ``bonus_malus``: 50, 51–57, 58–72, 73–90, 91–100, 101+ (eigene Wahl
      nach Volumen und Verlauf der Frequenz)
    * ``bm_off_ladder``: Wert liegt nicht auf der Leiter der schadenfreien
      Jahre ab 100, also nach einem Schaden (Malus mal 1,25). Nur eine
      Näherung: Nach weiteren schadenfreien Jahren kann ein Wert wieder auf
      der Leiter landen.
    * ``area``, ``veh_brand``, ``veh_gas``, ``region``: unverändert (Marke
      und Region in Zahlreihenfolge der Codes)

    Die Klassen sind geordnete Kategorien, damit Tabellen und Grafiken in
    natürlicher Reihenfolge erscheinen. Die Grenzen wurden auf allen Daten
    gewählt, nicht nur auf den Trainingsdaten; bei so groben Klassen ist der
    Vorteil daraus für die Testdevianz vernachlässigbar.

    Returns
    -------
    DataFrame mit dem Index von ``policies`` und den Spalten ``TARIFF_FEATURES``.
    """
    inf = np.inf
    power = policies["veh_power"].clip(upper=9)
    return pd.DataFrame(
        {
            "area": policies["area"],
            "veh_power": _cut(power, [4, 5, 6, 7, 8, 9, inf], ["4", "5", "6", "7", "8", "9+"]),
            "veh_age": _cut(policies["veh_age"], [0, 1, 11, inf], ["0", "1-10", "11+"]),
            "driv_age": _cut(
                policies["driv_age"],
                [18, 21, 26, 31, 41, 51, 71, inf],
                ["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"],
            ),
            "bonus_malus": _cut(
                policies["bonus_malus"],
                [50, 51, 58, 73, 91, 101, inf],
                ["50", "51-57", "58-72", "73-90", "91-100", "101+"],
            ),
            "bm_off_ladder": ~policies["bonus_malus"].isin(BONUS_MALUS_LADDER),
            "veh_brand": _numbered(policies["veh_brand"]),
            "veh_gas": policies["veh_gas"],
            "region": _numbered(policies["region"]),
        },
        index=policies.index,
    )


def duplicate_groups(policies: pd.DataFrame) -> pd.Series:
    """Gruppennummer je Police: gleiche Rohwerte aller Merkmale ergeben eine Gruppe.

    Vertragsverlängerungen im Lauf des Jahres stehen in freMTPL2 als
    getrennte Zeilen mit denselben Merkmalen (Wüthrich und Merz 2023). Eine
    Kennung, die sie verbindet, fehlt; gleiche Rohwerte sind die Näherung.
    Gebildet auf den Rohwerten, nicht auf den Klassen, sonst würden die
    Gruppen so groß, dass im Test ganze Merkmalskombinationen fehlen.
    Fremde Policen mit zufällig gleichen Merkmalen landen in derselben
    Gruppe; das macht die Aufteilung nur gröber, nicht ungültig.
    """
    return policies.groupby(list(RAW_FEATURES), sort=False).ngroup().rename("group")

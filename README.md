# Schadenmodellierung für ein Kfz-Portfolio

Dieses Projekt modelliert die Schäden eines simulierten Kfz-Portfolios in drei Teilen: Gesamtschaden im kollektiven Modell, Reservierung mit Chain-Ladder und Tarifierung mit Poisson- und Gamma-GLM; jede Schätzung wird gegen die bekannten wahren Parameter der Simulation geprüft. Das Frequenz-GLM läuft zusätzlich auf dem echten französischen Portfolio freMTPL2 und wird dort auf Testdaten bewertet. Es richtet sich an Leserinnen und Leser aus Aktuariat und versicherungsnaher IT.

## Ergebnisse auf einen Blick

Alle Zahlen und Grafiken stammen aus `nonlife run --seed 42` (Standardargumente: 10.000 Policen je Anfalljahr, Anfalljahre 2015–2024, Stichtag Ende 2024, 10.000 Monte-Carlo-Läufe). Der Lauf erzeugt 100.000 Policen, 7.192 Schäden und 39.426 Zahlungen. Die vollständigen Kennzahlen stehen in [`reports/results.json`](reports/results.json).

### Gesamtschaden: Simulation gegen Theorie

![Verteilung des Jahresgesamtschadens](reports/figures/aggregate_loss.png)

Die Monte-Carlo-Verteilung der Jahresschadenlast trifft den theoretischen Erwartungswert auf 0,04 % und die theoretische Standardabweichung auf 1,8 %.

| | Simulation | Theorie |
|---|---:|---:|
| $E[S]$ | 2.544.424 | 2.543.494 |
| $\mathrm{sd}[S]$ | 147.344 | 150.101 |
| 99-%-Quantil | 2.895.558 | – |

Parameter: $\hat\lambda = 719{,}2$ Schäden je Jahr, Schwere Lognormal mit $\hat\mu = 7{,}7118$, $\hat\sigma = 0{,}9582$.

### Reserve: Chain-Ladder gegen wahre Reserve

![Geschätzte gegen wahre Reserve je Anfalljahr](reports/figures/reserves.png)

Die Chain-Ladder-Reserve liegt insgesamt 1,3 % unter der wahren Reserve; je Anfalljahr reicht die Abweichung von −6,5 % (2020) bis +3,4 % (2018), nur das fast abgewickelte Jahr 2016 weicht bei sehr kleinem Betrag stark ab (7.678 gegen 4.723). Ein Mack-Intervall fehlt noch (siehe [Geplant](#geplant)).

| | Chain-Ladder | wahr | Abweichung |
|---|---:|---:|---:|
| Reserve gesamt | 3.326.580 | 3.371.937 | −1,3 % |

### Tariffaktoren: GLM gegen wahre Faktoren

![Frequenzfaktoren](reports/figures/relativities_frequency.png)

Das Poisson-GLM trifft die wahren Frequenzfaktoren; die einfache Auswertung je Merkmal ist für `vehicle_power` deutlich verzerrt (Klasse `high`: einfach 1,066, GLM 1,229, wahr 1,25), weil junge Fahrer mit hoher Frequenz überwiegend schwach motorisierte Fahrzeuge fahren und so die Referenzklasse `low` verteuern. Von den sieben geschätzten Frequenzfaktoren liegt bei einem der wahre Wert knapp außerhalb des 95-%-Intervalls: `driver_age: 60+` mit GLM 1,176, Intervall [1,103; 1,254], wahr 1,10. Bei sieben Intervallen zum Niveau 95 % ist das kein Widerspruch zum Modell.

![Schwerefaktoren](reports/figures/relativities_severity.png)

Das Gamma-GLM trifft die wahren Schwerefaktoren; alle sieben 95-%-Intervalle enthalten den wahren Wert (z. B. `vehicle_power: high`: GLM 1,393, wahr 1,40).

### Frequenz auf echten Daten: freMTPL2

Zahlen und Grafik aus `nonlife fremtpl2 --seed 42`, vollständig in [`reports/fremtpl2.json`](reports/fremtpl2.json). Nach der Bereinigung (siehe [Daten](#daten)) bleiben 678.007 Policen mit 26.383 Schäden. Aufgeteilt wird gruppenweise in 610.450 Policen zum Schätzen und 67.557 (10,0 %) zum Testen. Wahre Faktoren gibt es hier nicht; bewertet wird die Vorhersage auf den Testdaten.

| Poisson-Devianz je Police (×10⁻²) | Training | Test |
|---|---:|---:|
| Nullmodell (eine Frequenz für alle) | 25,2614 | 25,0110 |
| GLM, 9 Merkmale, 57 Parameter | 23,7987 | 23,5531 |

Das GLM senkt die Testdevianz um 5,83 % gegenüber dem Nullmodell. Das wirkt wenig, ist aber typisch: Bei einer Frequenz um 7 % ist der größte Teil der Streuung reiner Zufall, den kein Merkmal erklären kann.

![Ist gegen Soll je Klasse auf den Testdaten](reports/figures/fremtpl2_actual_vs_expected.png)

Auf den Testdaten liegen beobachtete und vorhergesagte Schadenzahl insgesamt bei 2.619 gegen 2.618,0. Die Balken zeigen die reine Poisson-Streuung ±1,96/√Soll; mit ihr liegen 6 der 65 Klassen außerhalb. Die Daten sind aber überdispers: Der Pearson-Schätzer auf den Trainingsdaten ist $\hat\phi = 1{,}71$. Mit um $\sqrt{\hat\phi}$ verbreiterten Bändern bleibt nur die Region R23 (Ist 34, Soll 20,8) außerhalb; bei 65 Klassen wären etwa drei zufällige Ausreißer zu erwarten. Aus demselben Grund sind die Konfidenzintervalle der Faktoren mit $\phi = 1$ um den Faktor $\sqrt{1{,}71} \approx 1{,}31$ zu schmal (siehe [Grenzen](#grenzen)).

Das stärkste Merkmal ist der Bonus-Malus. Neben seiner Höhe trägt das selbst gebildete Merkmal `bm_off_ladder` (Wert liegt nicht auf der Stufenleiter schadenfreier Jahre, also nach einem Schaden) einen Faktor von 2,19 [2,10; 2,29] bei gleicher Höhe.

## Methodik

### Kollektives Modell ([`collective.py`](src/nonlife/collective.py))

Jahresgesamtschaden $S = X_1 + \dots + X_N$ mit $N \sim \text{Poisson}(\lambda)$ und u.i.v. Schadenhöhen $X_k \sim \text{Lognormal}(\mu, \sigma^2)$, unabhängig von $N$. Dann gilt

$$E[S] = E[N]\thinspace E[X], \qquad \mathrm{Var}[S] = E[N]\mathrm{Var}[X] + \mathrm{Var}[N]\thinspace E[X]^2 = \lambda\thinspace E[X^2].$$

Das Portfolio ist heterogen: Police $i$ hat die Schadenanzahl $N_i \sim \text{Poisson}(\lambda_i)$ mit $\lambda_i = \lambda(x_i)\thinspace e_i$. Die Summe unabhängiger zusammengesetzter Poisson-Variablen ist wieder zusammengesetzt Poisson mit $\lambda = \sum_i \lambda_i$ und der Mischung $\sum_i (\lambda_i/\lambda)\thinspace F_i$ der Schwereverteilungen. Daher darf das Portfolio als ein kollektives Modell behandelt werden. Geschätzt wird $\hat\lambda$ als Schadenfrequenz je Exposure-Jahr mal mittlere Exposure eines Anfalljahres, $(\hat\mu, \hat\sigma)$ per Maximum-Likelihood aus allen Einzelschäden, also Mittelwert und Standardabweichung (Divisor $n$) der logarithmierten Schadenhöhen.

Die Monte-Carlo-Simulation ist vektorisiert: alle Schadenanzahlen in einem Aufruf, danach alle Schadenhöhen in einem Block, Summe je Lauf mit `np.bincount`. Zufall kommt ausschließlich aus einem übergebenen `numpy.random.Generator`.

### Reservierung ([`reserving.py`](src/nonlife/reserving.py))

Kumuliertes Zahlungsdreieck $C_{i,j}$ (Anfalljahr $i$, Abwicklungsjahr $j$) aus den Einzelzahlungen bis zum Stichtag. Volumengewichtete Entwicklungsfaktoren

$$\hat f_j = \frac{\sum_i C_{i,j+1}}{\sum_i C_{i,j}},$$

summiert über alle Anfalljahre, für die beide Werte beobachtet sind. Mit dem Stichtag $I$ ist $C_{i,I-i}$ der letzte beobachtete Wert des Anfalljahres $i$. Hochrechnung und Reserve:

$$\hat C_{i,J} = C_{i,I-i} \prod_{j \ge I-i} \hat f_j, \qquad \hat R_i = \hat C_{i,J} - C_{i,I-i}.$$

Die wahre Reserve je Anfalljahr ist der simulierte Endschaden minus die bis zum Stichtag gezahlten Beträge. Die Chain-Ladder-Funktionen sind gegen das Paket [`chainladder`](https://github.com/casact/chainladder-python) auf dem RAA-Dreieck aus Mack (1993) getestet; das Paket wird nur in den Tests verwendet.

### Tarifierung ([`pricing.py`](src/nonlife/pricing.py))

Frequenz: Poisson-GLM mit Log-Link und $\log e_i$ als Offset (Koeffizient fest 1, da $E[N_i]$ proportional zur Exposure),

$$\log E[N_i] = \log e_i + \beta_0 + \sum_m \beta_{m, x_{im}}.$$

Schwere: Gamma-GLM mit Log-Link auf den Einzelschäden, $\log E[X_k] = \gamma_0 + \sum_m \gamma_{m, x_{km}}$. Das Gamma-GLM unterstellt einen konstanten Variationskoeffizienten; das gilt für die simulierten lognormalen Schäden, sodass die Schätzer als Quasi-Likelihood-Schätzer konsistent sind. Die Dispersion wird über die Pearson-Statistik geschätzt.

Kategoriale Merkmale gehen mit Treatment-Kodierung ein, die Designmatrix wird selbst gebaut, damit die Referenzklasse frei wählbar ist. Tariffaktoren sind $e^{\hat\beta}$, die Konfidenzintervalle $\exp(\hat\beta \pm z_{1-\alpha/2}\thinspace \widehat{\mathrm{se}})$ (Wald-Intervall auf der Log-Skala, daher asymmetrisch um den Faktor). Zum Vergleich wird die einfache Auswertung je Merkmal $(N_\ell/E_\ell)/(N_\text{ref}/E_\text{ref})$ gezeigt; sie entspricht einem Poisson-GLM mit nur diesem Merkmal. Geschätzt wird mit `statsmodels`.

### Bewertung auf echten Daten ([`evaluation.py`](src/nonlife/evaluation.py))

Ohne wahre Parameter zählt die Vorhersage auf Policen, die das Modell nicht gesehen hat. Die Policen werden 90/10 in Trainings- und Testdaten geteilt, und zwar gruppenweise: Policen mit identischen Merkmalen bilden eine Gruppe und landen gemeinsam auf einer Seite (Begründung unter [Daten](#daten)). Referenz je Merkmal ist die Klasse mit der größten Exposure im Training. Maß ist die mittlere Poisson-Devianz auf den Testdaten,

$$D = \frac{1}{n} \sum_i 2 \left[ N_i \log \frac{N_i}{\hat\mu_i} - (N_i - \hat\mu_i) \right], \qquad \hat\mu_i = e_i \thinspace \hat\lambda(x_i), \quad 0 \cdot \log 0 = 0,$$

also das Kriterium, das das GLM beim Schätzen minimiert, ausgewertet auf neuen Daten. Vergleich ist das Nullmodell mit einer gemeinsamen Frequenz. Je Klasse wird $\sum N_i$ gegen $\sum \hat\mu_i$ verglichen; auf den Trainingsdaten sind beide nach der Score-Gleichung $X^\top (N - \hat\mu) = 0$ für jede Klasse im Modell gleich, aussagekräftig ist der Vergleich daher nur auf den Testdaten. Die Poisson-Annahme $\mathrm{Var}[N] = E[N]$ wird mit dem Pearson-Schätzer $\hat\phi = \frac{1}{n-p} \sum_i (N_i - \hat\mu_i)^2 / \hat\mu_i$ geprüft.

## Daten

- **Synthetisches Portfolio** ([`simulate.py`](src/nonlife/simulate.py)): Tarifmerkmale `driver_age`, `region` und `vehicle_power`, wobei `vehicle_power` bewusst von `driver_age` abhängt. Frequenz und mittlere Schadenhöhe sind multiplikativ in den Merkmalen, die wahren Faktoren sind bekannt (`default_parameters`). Schadenanzahl Poisson mit Exposure, Schadenhöhe lognormal mit festem Variationskoeffizienten. Jeder Schaden wird über zehn Abwicklungsjahre ausgezahlt; die Zahlungsanteile sind Dirichlet-verteilt um ein festes Abwicklungsmuster, damit die Entwicklungsfaktoren zwischen Anfalljahren streuen. Zahlungen nach dem Stichtag werden abgeschnitten und liefern die wahre Reserve.
- **freMTPL2** ([`fremtpl2.py`](src/nonlife/fremtpl2.py)): französisches Kfz-Haftpflicht-Portfolio eines Beobachtungsjahres aus dem R-Paket CASdatasets (Version 1.0-8), geladen von OpenML ([freMTPL2freq, ID 41214](https://www.openml.org/d/41214), eine Zeile je Police; [freMTPL2sev, ID 41215](https://www.openml.org/d/41215), eine Zeile je Schaden) und unter `data/openml/` zwischengespeichert.
  - **Bereinigung** nach Wüthrich und Merz (2023, Anhang B): Die Schadenzahl `ClaimNb` ist für viele Policen fehlerhaft (zu ihnen gibt es keine Schadenhöhen). Sie wird deshalb aus der Anzahl der Schadenhöhen neu berechnet; das ändert sie bei 9.117 Policen, davon 8.851 mit Policennummer ≤ 24.500. Keine Police wird dafür entfernt, sonst fielen nur Policen mit Schaden weg und die Frequenz sänke. Entfernt werden 6 Policen mit mehr als 5 Schäden (vermutlich Datenfehler) und 195 Schäden ohne Police; die Exposure wird bei 1 Jahr gekappt (1.224 Policen). Ergebnis: 678.007 Policen, 26.383 Schäden, wie bei Wüthrich und Merz.
  - **Klassen:** Fahrzeugstärke, Fahrzeug- und Fahreralter wie bei Noll, Salzmann und Wüthrich (2018). `Area` (A–F) ist im Datensatz bereits eine Einteilung der Bevölkerungsdichte (A: bis 50, F: über 10.000 Einwohner/km²); deshalb geht `Area` ein und `Density` nicht, beide zusammen wären fast kollinear. Der Bonus-Malus (französischer Schadenfreiheitsrabatt: Start bei 100, je schadenfreiem Jahr mal 0,95 bis zum Boden 50, je Schaden mal 1,25) geht als zwei Merkmale ein: seine Höhe in sechs Klassen und `bm_off_ladder`, ob der Wert neben der Stufenleiter 100, 95, 90, …, 51, 50 liegt, also ein Schaden in der Vergangenheit steckt.
  - **Gruppenweise Aufteilung:** Vertragsverlängerungen im Lauf des Jahres stehen als getrennte Zeilen im Datensatz (Wüthrich und Merz 2023). 38 % der Policen haben einen „Zwilling“ mit identischen Merkmalen; bei Paaren ist die Summe der Exposure in 96 % der Fälle höchstens 1 und der Abstand der Policennummern im Median 1. Bei einer Aufteilung je Zeile läge der Zwilling einer Testpolice meist im Training, und die Testdevianz wäre zu optimistisch.

  Quellen: M. V. Wüthrich, M. Merz (2023), *Statistical Foundations of Actuarial Learning and its Applications*, Springer, Anhang B ([Open Access](https://doi.org/10.1007/978-3-031-12409-9_13)); A. Noll, R. Salzmann, M. V. Wüthrich (2018), *Case Study: French Motor Third-Party Liability Claims* ([SSRN 3164764](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3164764)).

Erzeugte und heruntergeladene Daten liegen unter `data/` und werden nicht eingecheckt.

## Installation und Reproduktion

Python 3.11 oder neuer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

nonlife run                      # Seed 42; erzeugt data/, reports/results.json, reports/figures/
pytest
ruff check .
mypy src

pip install -e ".[fremtpl]"      # optional: scikit-learn zum Laden von freMTPL2
nonlife fremtpl2                 # lädt beim ersten Lauf von OpenML; reports/fremtpl2.json
pytest -m network                # Kontrollwerte der Bereinigung auf den echten Daten
```

Die Tests laufen ohne Netz und ohne scikit-learn; die Bereinigung wird dort an kleinen künstlichen Tabellen geprüft, in denen jede Police einen Fall abdeckt.

## Grenzen

- **Schwere im kollektiven Modell:** Für das gesamte Portfolio wird eine einzige Lognormalverteilung angepasst. Die wahre Schwere ist eine Mischung von Lognormalverteilungen über die Tarifklassen.
- **Zahlungs- statt Aufwandsdreieck:** Es gibt keine Spätmeldungen und keine Einzelfallreserven; jeder Schaden ist im Anfalljahr bekannt. Die Reserve ist hier die Summe der ausstehenden Zahlungen, kein IBNR im engeren Sinn.
- **Kein Tail-Faktor:** Das Abwicklungsmuster endet nach zehn Jahren und damit innerhalb des Dreiecks.
- **Keine Inflation, stationäres Portfolio:** Muster, Faktoren und Portfoliogröße sind über alle Anfalljahre gleich.
- **Synthetische, korrekt spezifizierte Daten:** Die GLMs schätzen genau die Modellklasse, aus der die Daten stammen. Fehlspezifikation, Interaktionen und Großschäden kommen nicht vor; deshalb läuft das Frequenzmodell zusätzlich auf freMTPL2.
- **Tarifierung auf Endschäden:** Die GLMs auf der Simulation verwenden die simulierten Endschäden aller Anfalljahre. In der Praxis wären die jungen Jahre noch offen; man würde den Aufwand auf Endschaden hochrechnen (etwa mit den Faktoren aus der Reservierung) oder nur abgewickelte Jahre verwenden.
- **freMTPL2, Überdispersion:** Das Frequenz-GLM rechnet mit $\phi = 1$, geschätzt ist $\hat\phi = 1{,}71$. Die Faktoren bleiben konsistent, ihre Konfidenzintervalle sind aber zu schmal. Quasi-Poisson oder Negativ-Binomial sind der nächste Schritt.
- **freMTPL2, ein Beobachtungsjahr:** Validiert wird auf anderen Policen desselben Jahres, nicht auf einem Folgejahr. Wie gut der Tarif in die Zukunft trägt, misst das nicht.
- **freMTPL2, Annahmen der Aufbereitung:** Die neu berechnete Schadenzahl unterstellt, dass eine Police ohne Schadenhöhe keinen Schaden hatte. Doppelt erfasste Verträge werden nur näherungsweise über gleiche Merkmale erkannt. Die Klassengrenzen wurden auf allen Daten gewählt, nicht nur auf den Trainingsdaten. Modelliert ist bisher nur die Frequenz; ob `ClaimAmount` abgewickelte Endschäden enthält, ist nicht dokumentiert.

## Geplant

- **Gradient Boosting als Messlatte für das GLM** (scikit-learn, Poisson-Verlust, Zielgröße $N/e$ mit Gewicht $e$ statt Offset), auf derselben Aufteilung: Wie viel Vorhersagekraft lässt das GLM liegen? Dabei auch die Aufteilung je Zeile zum Vergleich, um das Leck durch doppelt erfasste Verträge zu messen.
- **Wechselwirkungen finden und ins GLM übernehmen:** im Boosting-Modell nur auf den Trainingsdaten suchen (Partial Dependence, H-Statistik nach Friedman), als Merkmal ins GLM aufnehmen, auf den Testdaten prüfen.
- **Überdispersion:** Quasi-Poisson oder Negativ-Binomial für das Frequenz-GLM auf freMTPL2.
- **Schwere-GLM auf freMTPL2** mit Behandlung der Großschäden (Kappung oberhalb einer Schwelle, Überschuss auf alle Policen umgelegt).
- **Standardfehler der Reserve nach Mack (1993)** mit Intervall in der Reservegrafik, geprüft an Macks veröffentlichtem Beispiel und angewendet auf echte Dreiecke der CAS Loss Reserving Database, die auch die tatsächliche spätere Abwicklung enthält.
- **Panjer-Rekursion** mit diskretisierter Schwere, Vergleich der Quantile und der Laufzeit mit Monte Carlo.

## Entstehung

Für die fachliche Kernlogik hat Linus Plümer Signaturen, fachliche Annahmen und Tests festgelegt und freigegeben; implementiert wurde sie mit Claude Code; die gemeinsamen Reviews laufen. Bei freMTPL2 hat er die Entscheidungen zur Aufbereitung getroffen (Bereinigung, Klassen, Bonus-Malus als zwei Merkmale, gruppenweise Aufteilung, Bewertung über die Testdevianz). Projektstruktur, Tooling, CLI und Grafiken stammen weitgehend von Claude Code.

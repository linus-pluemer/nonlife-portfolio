# Schadenmodellierung für ein Kfz-Portfolio

Dieses Projekt modelliert Schäden eines Kfz-Portfolios in drei Teilen: Gesamtschaden im kollektiven Modell, Reservierung mit Chain-Ladder und Mack, Tarifierung mit Poisson- und Gamma-GLM. Jede Schätzung wird gegen die bekannten wahren Parameter eines simulierten Portfolios geprüft. Es richtet sich an Leserinnen und Leser aus Aktuariat und versicherungsnaher IT.

## Ergebnisse auf einen Blick

| | Grafik | Aussage |
|---|---|---|
| Gesamtschaden | *offen* | Verteilung des Jahresgesamtschadens, Monte Carlo gegen Theorie |
| Reservierung | *offen* | geschätzte gegen wahre Reserve je Anfalljahr, mit Mack-Intervall |
| Tarifierung | *offen* | geschätzte gegen wahre Tariffaktoren, mit Konfidenzintervall |

## Methodik

### Kollektives Modell

Jahresgesamtschaden $S = \sum_{i=1}^{N} X_i$ mit $N \sim \text{Poisson}(\lambda)$ und u.i.v. Schadenhöhen $X_i \sim \text{Lognormal}(\mu, \sigma^2)$, unabhängig von $N$. Die Parameter werden aus den Einzelschäden geschätzt: $\hat\lambda$ als Schadenanzahl je Exposure-Jahr, $(\hat\mu, \hat\sigma)$ per Maximum-Likelihood.

Die Monte-Carlo-Simulation liefert Erwartungswert, Varianz und 99-%-Quantil von $S$. Verglichen wird mit

$$E[S] = E[N]\,E[X], \qquad \operatorname{Var}[S] = E[N]\operatorname{Var}[X] + \operatorname{Var}[N]\,E[X]^2 .$$

Zusätzlich wird die Verteilung von $S$ per Panjer-Rekursion mit diskretisierter Schwere bestimmt (*offen*):

$$g_0 = e^{-\lambda(1-f_0)}, \qquad g_s = \frac{\lambda}{s}\sum_{k=1}^{s} k\, f_k\, g_{s-k}.$$

### Reservierung

Kumuliertes Dreieck $C_{i,j}$ (Anfalljahr $i$, Abwicklungsjahr $j$) aus den Einzelzahlungen. Volumengewichtete Entwicklungsfaktoren

$$\hat f_j = \frac{\sum_i C_{i,j+1}}{\sum_i C_{i,j}},$$

summiert über alle Anfalljahre mit beobachtetem $C_{i,j+1}$. Hochrechnung $\hat C_{i,J} = C_{i,I-i}\prod_{j \ge I-i}\hat f_j$, Reserve $\hat R_i = \hat C_{i,J} - C_{i,I-i}$. Standardfehler nach Mack (1993) (*offen*). Die Implementierung ist gegen das Paket [`chainladder`](https://github.com/casact/chainladder-python) auf dem RAA-Dreieck aus Mack (1993) getestet.

### Tarifierung (*offen*)

Frequenz: Poisson-GLM mit Log-Link und $\log(\text{Exposure})$ als Offset. Schwere: Gamma-GLM mit Log-Link. Beides mit `statsmodels`, kategoriale Merkmale mit Treatment-Kodierung. Die Tariffaktoren $e^{\hat\beta}$ werden mit Wald-Intervallen $e^{\hat\beta \pm z\,\widehat{\text{se}}}$ angegeben.

## Daten

- **Synthetisches Portfolio**: erzeugt von [`src/nonlife/simulate.py`](src/nonlife/simulate.py) mit festem Seed. Die wahren Frequenz- und Schwerefaktoren sind bekannt, Schätzungen lassen sich also direkt prüfen. Die Erweiterung um Anfalljahre und Zahlungsverlauf für die Reservierung ist *offen*.
- **freMTPL2**: französisches Kfz-Haftpflicht-Portfolio, frei verfügbar über OpenML ([freMTPL2freq, ID 41214](https://www.openml.org/d/41214), [freMTPL2sev, ID 41215](https://www.openml.org/d/41215)). Es wird bei Bedarf geladen und unter `data/` zwischengespeichert, aber nicht eingecheckt. Das Frequenzmodell auf diesen Daten ist *offen*.

## Installation und Reproduktion

Python 3.11 oder neuer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

nonlife run --seed 42            # erzeugt data/, reports/results.json, reports/figures/
pytest
ruff check .
mypy src
```

## Grenzen

- Das synthetische Portfolio ist bewusst einfach: multiplikative Effekte ohne Interaktionen, unabhängige Schäden, keine Inflation, keine Großschäden.
- Chain-Ladder setzt ein über alle Anfalljahre stabiles Abwicklungsmuster voraus. Trendbrüche bildet das Verfahren nicht ab.
- Das kollektive Modell behandelt das Portfolio als homogen und ignoriert die Merkmalsstruktur, die in der Tarifierung modelliert wird.

## Entstehung

Die fachliche Kernlogik (Monte-Carlo-Simulation, Chain-Ladder, Panjer, Mack, GLM, Simulation des Portfolios) schreibt Linus Plümer selbst. Claude Code wurde für Projektstruktur, Tooling, Datenladen, CLI, Grafikformatierung, das Gerüst dieser README sowie für Tests und Code-Reviews eingesetzt.

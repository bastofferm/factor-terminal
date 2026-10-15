# Timestamp-Review: Preise, Renditen, Zinsen und FX nach Region

**15. Oktober 2026.** Für diesen Review habe ich die Ingest- und Bau-Pipeline gelesen
(`sync.py`, `ingest_yahoo.py`, `ingest_fred.py`, `build_factors.py`,
`run_estimation.py`) und die Factors-DB sowie das Warehouse `xbrl_sec`
abgefragt. Jede Vermutung zum Timing habe ich an den Daten gemessen. Dafür gab es
zwei Prüfungen:

- **Lead-Lag gegen eine synchrone Referenz.** Für jede Reihe misst der Review die
  Korrelation mit einem Instrument, das zum NY-Schluss (16:00 ET) notiert, am
  selben Tag und am Vortag. Er schätzt außerdem die Regression
  a_t ~ b_{t−1} + b_t + b_{t+1}. Der **Lag-Anteil** ist β(b_{t−1}) / (β(b_t) + β(b_{t−1})):
  0 heißt synchron, 1 heißt einen Tag versetzt. Stichprobe: 2012 bis Mai 2026 auf
  dem Handelskalender des Modells.
- **Gespeicherte gegen heutige Schlusskurse.** Die Schlusskurse vom 15.09. bis
  08.10.2026 werden mit einem frischen Yahoo-Abruf verglichen. Ein Wert, der sich
  nachträglich geändert hat, war bei der Speicherung kein Schlusskurs.

Die Uhr des Modells ist der **NY-Schluss**. Fast alle Faktorinstrumente sind
NY-gelistete ETFs, und `ref_calendar` wird aus ihnen abgeleitet.

---

## Kurzfassung

| # | Befund | Schwere | Betroffen |
|---|---|---|---|
| 1 | Yahoo-FX (`…=X`) liegt **seit August 2010 einen ganzen Tag hinter** dem NY-Schluss, davor war es synchron | hoch | `fx_jpy`, `fx_carry`, `fact_fx` → `rt_uk`, USD-Umrechnung der JP-Aktien |
| 2 | Die FX-Tabelle endet am 09.07. bzw. 25.06.2026 und wird **ohne Limit vorwärts gefüllt** | hoch | `rt_uk` hat seit dem 09.07. kein GBP-Bein mehr, JP-Aktien haben seit Juni keine USD-Renditen |
| 3 | **Intraday-Stände werden als Schlusskurs gespeichert** und nie korrigiert | hoch | FX, `DX-Y.NYB`, `^VIX`, `IGLT.L` an jedem Lauftag; 214 falsche Renditen am 28.09.2026 |
| 4 | **Lücken:** ganze Handelstage fehlen, die Rendite landet auf dem Folgetag | mittel–hoch | 22.09.2026 fehlt im Kalender; `IGLT.L` fehlen drei Tage |
| 5 | **Ragged Edge:** Zinsfaktoren enthalten am Datenrand eine künstliche Nulländerung | mittel | `rt_us_*`, `rt_ea_level`, `rt_breakeven` am letzten Tag |
| 6 | Regionale Zinsfaktoren werden nur **kontemporär** orthogonalisiert | mittel | `rt_jp_level`, `rt_ea_level`, `rt_uk` |
| 7 | JP-Kalender: Renditen an US-Feiertagen fallen weg, JP-Feiertage erzeugen Mehrtagesrenditen | niedrig–mittel | JP-Einzeltitel |
| 8 | SOFR und EFFR erscheinen am Folgetag, gehen aber am Referenztag ein | niedrig | `liq_funding` |
| 9 | ANFCI und STLFSI4 tragen das Datum der Referenzwoche, nicht der Veröffentlichung | niedrig, latent | Rohdaten (in keinem Faktor) |

---

## Wie Zeit in der Datenbank dargestellt ist

- Jede Faktentabelle (`fact_input_return`, `fact_input_level`, `fact_input_fx`,
  `fact_factor_return`) speichert **nur ein `DATE`**: keine Uhrzeit, keine Zeitzone,
  kein Veröffentlichungszeitpunkt. Was ein Datum bedeutet, hängt implizit von der
  Quelle ab und steht nirgends in `ref_instrument` oder `ref_level_series`.
- Ausnahme ist `FRED:NFCI@release`, das unter seinem Veröffentlichungsdatum liegt.
  Das Warehouse hätte mit `sec.fact_macro_release.release_at` (TIMESTAMPTZ) eine
  Infrastruktur für Veröffentlichungszeitpunkte, das Projekt nutzt sie aber nicht.
- Der Handelskalender ist US-zentriert. Ein Datum zählt, wenn es ein Werktag ist und
  mindestens 30 % der Faktorinstrumente melden. Was auf einen Nicht-Kalendertag
  fällt, verschwindet beim `reindex`.

### Uhrzeit-Landkarte

Die Zeiten gelten für die Sommerzeit. Spalte „vs NY“ ist der Abstand zum NY-Schluss
um 20:00 UTC.

| Reihe | Quelle | Beobachtet um (UTC) | vs NY | Messung |
|---|---|---|---|---|
| US-ETFs: `SPY`, `IVV`, `ACWI`, `EWJ`, `VGK`, `EEM`, `TLT`, `HYG`, `LQD`, Sektoren, Rohstoffe | Yahoo / Warehouse | 20:00 | 0 | Referenzuhr. `EWJ` und `VGK` gegen `SPY`: Lag-Korrelation −0,07 bzw. −0,09, synchron |
| `DX-Y.NYB` | ICE via Yahoo | ≈ 20:00 | ≈ 0 | gegen `UUP` 0,96 am selben Tag, 0,02 am Vortag: synchron |
| `^VIX`, `FRED:VIXCLS` | Cboe | 20:15 | +¼ h | gegen `SPY` −0,75 am selben Tag: synchron |
| UST `FRED:DGS*`, `DFII10`, `T10YIE` | Treasury / FRED | ≈ 19:30 | −½ h | `DGS10` gegen `TLT` 0,90 / 0,02: synchron |
| EA-AAA-Kurve `ECB:BUND_*` | EZB, Referenztag | europäischer Nachmittag | einige Stunden früher | gegen UST10 0,55 am selben Tag, 0,22 am Vortag; Lag-Anteil 0,29 |
| JGB `MOF_JP:JGB_*` | MoF Japan | 06:00 (15:00 JST) | −14 h | gegen UST10 0,16 am selben Tag, **0,33 am Vortag**; Lag-Anteil 0,66 |
| `IGLT.L` (Gilts, GBP) | LSE via Yahoo | 15:30 | −4½ h | gegen `TLT` 0,57 / 0,15; Lag-Anteil 0,23 |
| JP-Aktien `fact_prices_jp` | Tokio | 06:00 / 06:30 | −14 h | 7203, 6758, 8306 gegen `SPY`: 0,13–0,14 am selben Tag, **0,29–0,31 am Vortag** |
| **Yahoo-FX `…=X` und Warehouse `fact_fx`** | Yahoo | **≈ 23:00 am Vortag** (Mitternacht London) | **≈ −21 h** | siehe Befund 1 |
| SOFR, EFFR, OBFR, TGCR | NY Fed | Geschäftstag t, publiziert t+1 um 12–13 UTC | **+16 h** | Befund 8 |
| `FRED:DFF` (Cash) | FRED | Kalendertage inkl. Wochenende | geht mit 1 Tag Lag ein | korrekt |
| ICE BofA OAS `FRED:BAML*` | ICE / FRED | Referenztag; Monatsenden auch an Wochenenden | – | gegen `HYG` −0,44 am selben Tag, −0,28 am Vortag (träge Anleihebewertung); in keinem Faktor |
| `FRED:NFCI@release` | ALFRED | Veröffentlichungstag (Mittwoch) | – | korrekt |
| `FRED:NFCI`, `ANFCI`, `STLFSI4` | FRED | **Freitag der Referenzwoche** (100 % Freitage) | Veröffentlichung 5–6 Tage später | Befund 9 |
| `ECB:CISS_EA_NEW` | EZB | Referenztag, acht vereinzelte Wochenenddaten | – | in keinem Faktor |
| `ECB:DFR` | EZB | Wirksamkeitsdatum, rund eine Woche nach dem Beschluss | konservativ | korrekt |
| `BOJ:IR01_OCRT` | BoJ | täglich | geht mit `shift(1)` ein | korrekt |

---

## Befunde

### 1. Yahoo-FX ist seit August 2010 um einen Tag versetzt (hoch)

Der Wert, den Yahoo als Tages-„Close“ eines FX-Paars unter dem Datum t führt, ist der
Kurs um **Mitternacht London zu Beginn von t**. In der Sommerzeit ist das 23:00 UTC
am Vortag, rund drei Stunden nach dem NY-Schluss von t−1. Die Stundenbalken
bestätigen das. Bei historischen Tagesbalken gilt Open ≈ Close
(EURUSD 08.10.: 1,1201 / 1,1201), und der Stundenbalken, der dem Close am nächsten
kommt, liegt jeweils um 23:00 UTC des Vortags.

Gemessen gegen die NY-gelisteten Währungs-ETFs (Korrelation der Tagesrenditen):

| Zeitraum | `USDJPY=X` gegen −`FXY`, selber Tag | Vortag | `EURUSD=X` gegen `FXE`, selber Tag | Vortag |
|---|---|---|---|---|
| 2009 | +0,94 | +0,07 | +0,96 | −0,03 |
| bis Juli 2010 (monatlich) | +0,81 … +0,99 | ≈ 0 | | |
| ab August 2010 (monatlich) | ≈ 0 | +0,85 … +0,96 | | |
| 2011–2026 gesamt | +0,06 | | +0,04 | |
| 2011–2026, **auf den Vortag umdatiert** | **+0,91** | | **+0,93** | |

`GBPUSD=X` gegen `FXB` verhält sich genauso (0,09 wie gespeichert, 0,93 umdatiert).
Für 2008–2010 verschlechtert die Umdatierung die Korrelation. Die Konvention ist
also **in der Reihe selbst gebrochen**, und eine pauschale Verschiebung um einen Tag
repariert sie nicht.

Was davon betroffen ist:

- **`fx_jpy`** und **`fx_carry`** (`EURUSD=X`, `USDJPY=X`, `USDCHF=X`) messen die
  FX-Bewegung des Vortags. `fx_jpy` gegen `eq_global` korreliert am selben Tag mit
  +0,01, mit dem Vortag mit −0,15. Die Risk-off-Bindung des Yen ist in der täglichen
  Kovarianz damit unsichtbar.
- **Die Orthogonalisierung greift nicht.** `fx_jpy` wird gegen `fx_usd`
  (`DX-Y.NYB`, synchron) residualisiert. Nach der Orthogonalisierung korreliert es
  mit `fx_usd` am selben Tag mit +0,03 und mit `fx_usd` vom Vortag mit **−0,29**:
  der USD-Anteil steckt noch im Faktor.
- **Dimson-Lags helfen hier nicht.** `dimson_design` legt verzögerte Faktoren an,
  passend zu einem Wertpapier, das vor dem Faktor schließt. Hier schließt der Faktor
  vor dem Wertpapier. Dafür bräuchte es einen *Lead* X_{t+1}, und das Projekt sieht
  bewusst keinen vor. Der Lead-Lag-Term der h-Tages-Kovarianz (Γₗ + Γₗ′) fängt einen
  Teil davon auf, die tägliche Prognose und die Betas nicht.
- **`fact_fx` im Warehouse** kommt aus derselben Quelle (`{CCY}USD=X`, siehe
  `xbrl_sec/etf/fx.py`). Daran hängen `Panels.in_base_ccy`, also `rt_uk`, und die
  USD-Spalten von `fact_prices_jp`. Eine Ausnahme gibt es: Für Tokio liegt der
  FX-Stempel (≈ 08:00 JST) nur sieben Stunden vor dem Börsenschluss und passt damit
  zufällig besser als für jede NY-Reihe.
- Nicht betroffen sind `fx_usd` (`DX-Y.NYB`), `fx_em` (`CEW`, ein ETF) und die
  ARP-Universen, die `FXE` und `FXY` als ETFs nutzen.

**Empfehlung:** FX für die Faktoren aus einer Quelle mit NY-Schluss nehmen, etwa
`FXY`, `FXE`, `FXB` und `FXF` als Spot-Proxy, deren Zinskomponente bei Bedarf
heraus­gerechnet wird. Alternativ die Yahoo-Werte ab dem 01.08.2010 auf den vorigen
Handelstag umdatieren und davor unverändert lassen. Ein Test sollte das absichern:
Für jedes `=X`-Paar muss die Korrelation mit seinem ETF am selben Tag größer sein als
am Vortag.

### 2. Die veraltete FX-Tabelle wird unbegrenzt vorwärts gefüllt (hoch)

`fact_input_fx` endet bei EUR, GBP und CHF am **09.07.2026**, bei JPY am
**25.06.2026**. Der Warehouse-FX-Job ist seitdem nicht gelaufen, während die Renditen
bis zum 08.10. reichen. `load_panels` füllt mit `fx.reindex(idx).ffill()` ohne Limit
auf. Von da an ist die FX-Rendite exakt null:

| `rt_uk` gegen `IGLT.L` lokal (ab März 2026) | Korrelation | Korrelation FX-Bein mit `GBPUSD=X` | Std.-Abw. FX-Bein |
|---|---|---|---|
| bis 09.07. | 0,81 | 0,92 | 0,0049 |
| nach 09.07. | **1,0000** | 0,07 | **0,000004** |

`rt_uk` ist seit drei Monaten ein reiner GBP-Gilt-Faktor, der sich als USD-Faktor
ausgibt. Der Liveness-Gate in `refresh_coverage` prüft nur `fact_input_return`, nicht
`fact_input_fx`. Dazu passt: In `fact_prices_jp` ist `fx_rate_usd_per_unit` ab Juni
2026 leer. Weil `sync_security` `log_return_usd IS NOT NULL` verlangt, enden alle
JP-Einzeltitel stillschweigend im Juni.

**Empfehlung:** In `load_panels` ein `ffill(limit=…)` setzen, einige Tage reichen,
und `fact_input_fx` in den Liveness-Gate aufnehmen. Die Umrechnung sollte aus den
täglich aufgefrischten `=X`- oder ETF-Reihen in `fact_input_return` kommen statt aus
dem Warehouse.

### 3. Intraday-Stände werden als Schlusskurs gespeichert und nie korrigiert (hoch)

Der Scheduler ist für 23:00 lokal ausgelegt. Laut `etl_run` liefen die Ingests aber
morgens oder nachmittags: am 24.09. um 08:49, am 25.09. um 09:50, am 28.09. um
**18:28 MESZ (12:28 ET, mitten in der US-Sitzung)**, am 29.09. um 07:45 und am 08.10.
um 09:30. Yahoo liefert für eine laufende Sitzung einen unfertigen Tagesbalken.
`ingest_yahoo` speichert ihn, und der nächste inkrementelle Lauf beginnt bei
`max(date) + 1`. Der Wert wird also **nie wieder überschrieben**.

Gespeichert gegen heute bei Yahoo, 15.09. bis 08.10.:

| Instrument | Abweichende Tage | größte Abweichung |
|---|---|---|
| `USDJPY=X` | 6 von 18 | −0,48 % (25.09.) |
| `EURUSD=X` | 6 von 18 | +0,14 % |
| `DX-Y.NYB` | 6 von 18 | +0,21 % |
| `^VIX` | 3 von 18 | **+5,0 %** (25.09.) |
| `IGLT.L` | 3 von 15 | +0,47 % (22.09.) |

Die Abweichungen liegen genau auf den Lauftagen. US-ETFs sind nicht betroffen, weil
ihre Sitzung bei Morgenläufen noch nicht begonnen hatte.

Am **28.09.2026** kommt ein zweiter Effekt hinzu. Bei **214** Yahoo-Zeilen ist
`ret_log ≠ ln(adj_t / adj_{t−1})`. Der Folgelauf hat den Schlusskurs nachgeladen,
die Rendite aber gegen den gespeicherten Intraday-Stand gerechnet
(`prev_adj` = letzte gespeicherte Zeile). Gespeichert ist damit die Rendite von
12:28 ET bis Handelsschluss, nicht die Tagesrendite. Beispiele: `^VIX` −0,87 % statt
+2,90 %, `ACWI` −0,52 % statt −0,85 %, `DBC` +0,49 % statt −0,40 %. Bis 2025 gibt es
keine einzige solche Inkonsistenz.

**Empfehlung:** Jeder inkrementelle Lauf lädt ein Überlappungsfenster neu, etwa die
letzten fünf Handelstage, und überschreibt es. Die Renditen werden danach aus den
gespeicherten `adj_close` neu berechnet statt aus dem batch-lokalen `prev_adj`.
Zusätzlich sollte ein Balken erst gespeichert werden, wenn die Börse des Instruments
geschlossen hat: Yahoo liefert die Börsenzeitzone mit, dazu kommt ein Puffer. Eine
Prüfung `ret_log = ln(adj/adj_prev)` gehört in `run_diagnostics`.

### 4. Lücken: ganze Handelstage fehlen (mittel–hoch)

Für **Dienstag, den 22.09.2026** (US-Handelstag) liegen nur 34 von rund 142
Faktorinstrumenten vor, darunter `SPY`, `TLT`, `IWM` und `GLD`. `ACWI`, `EWJ`, `HYG`
und rund hundert andere fehlen. Weil das Quorum von 30 % verfehlt ist, streicht
`trading_calendar` den Tag. Das hat zwei Folgen:

- Die Rendite vom 22.09. für `SPY` und die anderen 33 Instrumente ist **verloren**.
  Ihre Rendite vom 23.09. umfasst nur einen Tag.
- Für `ACWI`, `EWJ`, `HYG` und die übrigen umfasst die Rendite vom 23.09. **zwei
  Tage**. `eq_global` am 23.09. (−1,03 %) ist eine Zweitagesrendite.

`IGLT.L` fehlen der 21.09., der 24.09. und der 07.10., obwohl Yahoo diese Tage führt.
Die Ursache ist dieselbe Wasserstandslogik wie in Befund 3: Ein Datum, das in einem
Batch fehlt, wird übersprungen, sobald ein späteres Datum gespeichert ist.

**Empfehlung:** Dasselbe Überlappungsfenster wie in Befund 3 löst das. Zusätzlich
sollte eine Lückenprüfung gegen den Börsenkalender laufen. Ein Kalendertag, der am
Quorum scheitert, obwohl er ein Werktag ist, gehört in eine Warnung und darf nicht
still verschwinden.

### 5. Ragged Edge: Nullrenditen am Datenrand (mittel)

Beim letzten Build am 13.10. reichten die Renditen bis zum 07./08.10., die
FRED-Zinsen und die EZB-Kurve bis zum 06.10. und die JGB bis zum 07.10. Die
Kurvenbauer füllen mit `ffill(limit=5)` auf, deshalb ist die Zinsänderung am 07.10.
null:

- `rt_us_level` am 07.10. = 0,000052. Das ist nur der Carry, die Zinsänderung ist
  künstlich null.
- `rt_uk` endet am 06.10. (Lücke bei `IGLT.L`, Befund 4).

Weil `build_factors` immer komplett neu baut, verschwindet die Null beim nächsten
Build. Jede Schätzung, jeder Risk-Run und jede Agentenentscheidung dazwischen
behandelt sie aber als echte Beobachtung, genau am aktuellsten Tag, der für eine
Prognose am meisten zählt.

**Empfehlung:** Das Faktorpanel am gemeinsamen As-of-Datum abschneiden, also beim
Minimum der letzten echten Beobachtung über alle Blöcke. Alternativ wird
`ffill(limit=…)` nur *innerhalb* einer Reihe angewendet, nie über ihr letztes
beobachtetes Datum hinaus.

### 6. Regionale Zinsfaktoren werden nur kontemporär orthogonalisiert (mittel)

`rt_ea_level`, `rt_jp_level` und `rt_uk` werden gegen `rt_us_level` vom selben Tag
residualisiert. Der Teil der US-Bewegung, der erst nach dem europäischen oder
Tokioter Schluss entsteht, kommt aber erst am Folgetag an. Nach der
Orthogonalisierung bleibt er vollständig im Faktor:

| Faktor (`ret_orth`) | Korrelation mit `rt_us_level` am selben Tag | am Vortag | Lag-Anteil |
|---|---|---|---|
| `rt_jp_level` | +0,03 | **+0,35** | 0,91 |
| `rt_ea_level` | −0,02 | **+0,26** | 1,05 |
| `rt_uk` | −0,01 | **+0,20** | 1,02 |
| `eq_japan` (zum Vergleich) | 0,00 | +0,02 | – |

Die Aktienfaktoren sind hier sauber, weil sie aus NY-gelisteten ETFs gebaut sind.
Diese Designentscheidung ist richtig und sollte bleiben. Für die Zinsfaktoren
existieren zwei Korrekturen: Dimson-Lags in der Schätzung und der Lead-Lag-Term der
h-Tages-Kovarianz (Methodology Review §8). Die Faktoren selbst bleiben trotzdem
nicht „ex-US“.

**Empfehlung:** In die `orth`-Liste dieser drei Faktoren zusätzlich `rt_us_level`
vom Vortag aufnehmen. Das ist kausal, denn die Information liegt um t vor. Gemäß der
Messdisziplin des Projekts gilt die Änderung erst, wenn sie auf den Hold-out-Daten
und auf Portfolioebene gegen das eigene EWMA gewinnt.

### 7. JP-Einzeltitel auf dem US-Kalender (niedrig–mittel)

`estimate` richtet die Faktoren mit `factors.reindex(y.index)` auf die Daten des
Wertpapiers aus. Für JP:7203 im Zeitraum 2012–2025:

- **120** JP-Renditen liegen auf US-Feiertagen. Dort fehlen die Faktoren, die Zeile
  fällt weg, und ihre Rendite (3,3 % der Varianz) ist verloren.
- **194** US-Handelstage haben keine JP-Rendite. Nach einem JP-Feiertag, etwa in der
  Golden Week, trägt die nächste JP-Rendite mehrere Tage, wird aber gegen
  Eintagesfaktoren regressiert.

**Empfehlung:** Wertpapierrenditen auf den Faktorkalender aggregieren. Log-Renditen
von Nicht-Kalendertagen werden zum nächsten Kalendertag addiert, und über JP-Lücken
werden die Faktorrenditen summiert.

### 8. Geldmarktsätze werden am Folgetag veröffentlicht (niedrig)

`liq_funding` (SOFR − EFFR) nutzt den Satz für Geschäftstag t am Tag t. Die NY Fed
veröffentlicht SOFR aber erst an t+1 um etwa 08:00 ET und EFFR um etwa 09:00 ET,
rund 16 Stunden nach dem NY-Schluss. Messbar ist der Effekt nicht: Die Korrelation
mit `SPY` liegt bei jedem Lag unter 0,03. Das As-of-Prinzip ist trotzdem verletzt.
Der Cash-Satz `DFF` (mit Lag in `to_excess_return`) und die Carry-Differenz
(`shift(1)`) sind korrekt.

**Empfehlung:** `liq_funding` um einen Tag verschieben oder wie NFCI
release-getimt einspielen.

### 9. Wöchentliche Stressindizes mit Referenzdatum (niedrig, latent)

`FRED:ANFCI`, `FRED:STLFSI4` und die FRED-Kopie `FRED:NFCI` liegen ausnahmslos auf
dem Freitag der Referenzwoche. Veröffentlicht werden sie Mittwoch bzw. Donnerstag
danach. Kein Faktor nutzt sie, aber sie sind aktiv registriert, im Raw-Explorer
sichtbar und als Baustein greifbar. Wer daraus einen Faktor baut, holt sich fünf bis
sechs Tage Look-ahead, genau den Fehler, den `NFCI@release` behoben hat.

**Empfehlung:** Beide wie NFCI über ALFRED auf den Veröffentlichungstag legen, oder
in `ref_level_series` als `reference_dated` markieren und für Faktoren sperren.

### Kleinere Beobachtungen

- **`SNB:POLICY_RATE` hat 0 Zeilen.** Das CHF-Bein fällt in `fx_carry` stillschweigend
  weg (`continue`). Die „G4-Approximation“ besteht damit aus zwei Beinen, EUR und JPY.
  Das ist kein Timing-Fehler, ist aber beim Review aufgefallen.
- **`^VIX` aus Yahoo und `VIXCLS`** weichen bis zu 0,17 Log-Punkte voneinander ab, das
  stammt aus den Intraday-Ständen in Befund 3. Die Varianzprämie nutzt `VIXCLS` und
  ist davon nicht betroffen.
- **ICE BofA OAS:** FRED liefert die Historie nur ab Mai 2023, Monatsenden stehen
  auch auf Wochenenden (12 Daten). OAS-Änderungen hängen `HYG` teilweise einen Tag
  hinterher, weil die Anleihebewertung träge ist. Kein Faktor nutzt sie.
- **JGB-Samstagsdaten** vor 1989 (70 bis 519 Zeilen je Laufzeit) liegen außerhalb des
  Samples und fallen über den Kalender heraus. Sie sind harmlos.
- **ECB CISS** hat acht vereinzelte Wochenenddaten. Harmlos, weil die Reihe nicht
  verwendet wird.

---

## Was korrekt ist und bleiben sollte

- Regionale Aktienfaktoren kommen aus **NY-gelisteten ETFs** (`EWJ`, `VGK`, `EEM`) und
  sind damit synchron zu allen anderen Faktoren.
- US-Zinsen, Breakeven, `VIXCLS` und `DX-Y.NYB` sind synchron zum NY-Schluss.
- Duration und Konvexität werden bei y_{t−1} ausgewertet. Der Cash-Satz geht mit Lag
  ein, die Carry mit `shift(1)`. Die Momentum-Signale nutzen Daten bis t−1, die
  Varianzprämie VIX_{t−1}.
- `NFCI@release` liegt auf dem Veröffentlichungstag. `roll_onto_calendar` schiebt
  Releases an Feiertagen auf den nächsten Handelstag, statt sie zu verwerfen.
- Niederfrequente Reihen werden als Release-Ereignis geführt und nicht vorwärts
  gefüllt.

---

## Empfehlungen in Reihenfolge

1. **FX-Quelle reparieren** (Befunde 1 und 2): FX mit NY-Schluss für die Faktoren,
   Strukturbruch im August 2010 beachten, `ffill` begrenzen, `fact_input_fx` in den
   Liveness-Gate.
2. **Inkrementelle Ingests mit Überlappungsfenster** (Befunde 3 und 4): die letzten
   N Handelstage neu laden, Renditen aus gespeicherten Kursen berechnen, nur
   geschlossene Sitzungen speichern. Danach die Yahoo-Instrumente einmal voll neu
   laden (`--full`), um den 22.09., den 28.09. und die Intraday-Stände zu
   reparieren.
3. **Gemeinsames As-of-Datum** für das Faktorpanel (Befund 5).
4. **Zeitsemantik explizit machen:** additive Spalten wie `close_tz`,
   `obs_time_local` und `publication_lag_days` in `ref_instrument` und
   `ref_level_series`, weil die DB mit den Worktrees geteilt ist. Die Pipeline kann
   Abstände zum NY-Schluss dann prüfen, statt sie vorauszusetzen.
5. **Regionale Zinsfaktoren** zusätzlich gegen `rt_us_level` vom Vortag
   orthogonalisieren (Befund 6), gemessen auf den Hold-out-Daten und auf
   Portfolioebene.
6. **Kleinere Korrekturen:** `liq_funding` verschieben, ANFCI und STLFSI4
   release-timen oder sperren, JP-Renditen auf den Faktorkalender aggregieren,
   SNB-Satz beschaffen.
7. **Regressionstests:** FX-Lead-Lag gegen ETF, `ret_log = ln(adj/adj_prev)`, kein
   gespeicherter Balken vor Börsenschluss, keine Werktagslücke ohne Feiertag.

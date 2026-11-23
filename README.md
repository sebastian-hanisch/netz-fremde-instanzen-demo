# Das Netz auf anderen Rundtouren – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-netz-fremde-instanzen-demo.streamlit.app/)** -->

Fünfte Demo der Linie **Neuronale kombinatorische Optimierung erklärt** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie nimmt zwei Netze aus den Vorgängern der Linie, das **Zeigernetz** (12 480 Parameter, `zeigernetz-demo`) und das **Merkmalsnetz K** (145 Parameter, `zugwahl-lernen-demo`), die beide nur bei 20 gleichverteilten Orten trainiert wurden, und prüft sie auf Rundtouren, die anders aussehen: kleiner oder größer gezeichnet, verschoben, mit mehr Orten, in Gruppen oder auf einem Gitter. Vehikel ist die Rundtour (TSP) mit erzeugten Orten. In dieser Reihe prüft diese Demo Netze außerhalb ihrer Trainingsverteilung systematisch (Skala und Lage, fünf Größen, drei Familien); `rechnen-beim-einsatz-demo`, `zeigernetz-demo` und `zugwahl-lernen-demo` zeigen das nur als Ausblick bei n = 50. Ein Neuheitsanspruch über die Reihe hinaus besteht nicht (Literatur siehe unten).

## Kernfrage und Befund in Kürze

**Wechselt man nur Skala oder Lage der Instanz, bricht das Zeigernetz mit rohen Koordinaten als Eingabe in 14 von 18 Zellen unter den Nächsten Nachbarn; mit normierter Eingabe (beim Einsatz oder auch im Training) und beim Merkmalsnetz mit der Skala s der Instanz bricht es in 0 von 18 Zellen, weil die Netze dieselbe Eingabe sehen (Konstruktion, kein Befund über Lernen); das Merkmalsnetz mit fester Skala bricht in 4 von 18 Zellen. Wechselt man Größe oder Verteilung, bricht das Zeigernetz mit roher Eingabe bei n = 50 geclustert und in 5 von 6 Zellen ab n = 100, mit normierter Eingabe in 5 von 6 Zellen ab n = 100, das Merkmalsnetz in 0 von 15 Basiszellen.** Voraussetzungen: erzeugte Rundtouren (drei Familien), Training nur bei n = 20 gleichverteilt, fünf Zufalls-Seeds je Netz, eine greedy-Tour je Instanz, 50 bis 100 Instanzen je Zelle, eigene kleine Netze von Hand in numpy, Latte „Nächster Nachbar“.

**Das ist eine niedrige Latte.** Der Nächste Nachbar mit 2-opt liegt bei n = 50 gleichverteilt bei 5,00 % Lücke, das Merkmalsnetz bei 11,35 bis 16,29 %, das Zeigernetz bei 19,62 bis 19,85 %. „Hält den Nächsten Nachbarn“ heißt nicht „ist gut“. Ob die Brüche des Zeigernetzes an seiner Eingabe (Koordinaten) oder an seiner Bauweise oder am Training bei einer einzigen Größe hängen, ist durch den Vergleich beider Netze **nicht** entschieden; Unterschiede gelten für diese beiden Netze, nicht für „Attention“ als Ursache (Experiment 5 prüft nur das Training).

* **Skala und Lage** (Experiment 1). Faktor 10, n = 20 und n = 50 gleichverteilt: das Zeigernetz A hat 136,86 und 281,99 % Lücke (Nächster Nachbar 16,48 und 20,90 %); in allen 18 Wechselzellen liegt seine Lücke um mindestens 2 Prozentpunkte über der seiner Basiszelle. Dagegen bleiben A+nt, An und B mit s in allen 18 Zellen genau auf der Basiszelle (Anteil gleicher Touren 1,00, Konstruktion). Das Merkmalsnetz ohne s bricht bei Faktor 10 in 4 von 6 Zellen und bei Faktor 0,25 und Verschiebung in 0 von 12. Diagnosen (Hinweise, keine Ursachenmessung): die Zeiger-Werte von A (Betrag über 9,64 von 10, also tanh über 0,964) sind bei Faktor 10 zu 99,0 % im Sättigungsbereich (30 Instanzen, n = 20 gleichverteilt, Mittel über fünf Zufalls-Seeds; Basis 47,6 %, Faktor 0,25 0,1 %, Verschiebung 80,7 % bei einer Spanne der Zufalls-Seeds von 45 bis 99 %); beim Merkmalsnetz ohne s sind bei Faktor 10 98,5 % der Entfernungsmerkmale x1 gekappt (Basis 20,7 %).
* **Kosten der Normierung** (Experiment 1 und 2): der größte Betrag des Lückenunterschieds A+nt oder An gegen A auf den 15 Basiszellen ist 2,20 Prozentpunkte (fünf Zufalls-Seeds, 50 bis 100 Instanzen je Zelle); ob das mehr ist als die Streuung über Zufalls-Seeds, ist nicht geprüft. „Ohne Kosten“ gilt exakt nur per Konstruktion auf den Wechselzellen.
* **Größe und Verteilung** (Experimente 2 und 3). Zeigernetz A: bricht in 6 von 15 Basiszellen (n = 50 geclustert, ab n = 100 in allen Familien außer Gitter bei n = 100); A+nt und An je in 5 von 15 (n = 50 geclustert nur „nicht unterscheidbar“: +0,78 [−0,47; +2,01] beziehungsweise +0,81 [−0,51; +2,08]). Merkmalsnetz: B mit s und B ohne s schlagen den Nächsten Nachbarn in 15 von 15 Basiszellen, bei n = 100 gleichverteilt −3,04 [−4,49; −1,64] beziehungsweise −8,95 [−10,19; −7,76] Prozentpunkte, bei n = 200 gleichverteilt (50 Instanzen) −3,74 beziehungsweise −6,77. Das Merkmalsnetz ohne s ist bei n = 50 und n = 100 gleichverteilt besser als mit s (Lücke 11,35 gegen 16,29 % und 11,18 gegen 17,08 %); die Ursache ist nicht untersucht (s der Instanz geteilt durch s0 sinkt von 1,42 bei n = 10 auf 0,30 bei n = 200, ein Hinweis, keine Erklärung).
* **Instanzzahl** (Experiment 4). Aus den 100 Instanzen der Zelle n = 100 gleichverteilt kommt mit je 10 Instanzen für B mit s nur in 69 von 200 Teilstichproben dasselbe Urteil „schlägt“ heraus wie mit allen 100; bei A bei n = 200 gleichverteilt (50 Instanzen) in 200 von 200 das Urteil „bricht“ (halbe Intervallbreite im Mittel 3,2 Prozentpunkte).
* **Mischtraining** (Experiment 5). Das Zeigernetz mit normierter Eingabe und Training bei n = 10, 20 und 50 (Amix) bricht wie A+nt in 5 von 15 Basiszellen und in 5 von 6 Zellen ab n = 100, bei n = 200 in 3 von 3 Zellen (n = 200 gleichverteilt +11,90 gegen +12,67 bei A+nt); bei n = 50 gleichverteilt und bei n = 100 Gitter schlägt es den Nächsten Nachbarn (−1,98 und −2,08), A+nt liegt dort bei −1,28 und −0,23, nicht unterscheidbar vom Nächsten Nachbarn; ein Unterschied zwischen Amix und A+nt ist nicht geprüft. Das Mischtraining bei n = 10, 20 und 50 rettet n ≥ 100 in dieser Rezeptur also nicht, das spricht gegen „nur das Training bei einer einzigen Größe“; ob andere Rezepte oder größere Trainingsgrößen es ändern, ist nicht gemessen. Das Netz mit Augmentierung (Aaug) lernte in dieser Rezeptur nicht: 0 von 5 Zufalls-Seeds liegen auf der Trainingszelle unter dem Nächsten Nachbarn (Lücke 82,69 % bei n = 20 gleichverteilt); seine 18 Brüche bei Skala und Lage sagen nichts über Augmentierung im Allgemeinen (andere Lernrate oder mehr Schritte sind nicht gemessen).
* **Kosten** (K und O nebeneinander): eine Konstruktion kostet für den Nächsten Nachbarn und beide Netze n(n−1)/2 Kandidatenbewertungen (190 bei n = 20); in O kostet das Zeigernetz bei n = 20 das 162-Fache (246 384 gegen 1 520), das Merkmalsnetz das 31-Fache (47 242, mit s 47 642). K begünstigt die Netze, O benachteiligt sie; Training als dritter Posten (960 000 gesampelte Touren je Netz).

## Warum dieses Problem

Die Netze der Linie wurden bisher fast nur dort geprüft, wo sie trainiert wurden (bei 20 und 10 gleichverteilten Orten); einzelne Ausblicke bei n = 50 gibt es in den Vorgängern. Ein Netz, das Rundtouren baut, sieht aber in der Praxis Instanzen, die anders aussehen. Zwei Dinge sind dabei zu trennen: derselbe Plan, nur **anders gemessen** (Skala und Lage der Koordinaten; hier hilft eine Zeile Normierung der Eingabe, und die Gleichheit folgt aus der Konstruktion) und **wirklich andere Instanzen** (mehr Orte, Gruppen, Gitter; hier hilft die Normierung nicht). Die Demo misst beides getrennt und nimmt als Latte den Nächsten Nachbarn: eine niedrige Latte, was sie im Ergebnis ausdrücklich sagt.

## Modell

Rundtour (TSP) mit n Orten in der Ebene, euklidische Entfernungen; Instanzen der Quelle `hyperheuristik-demo` (über die Kopie in `zeigernetz-demo`), im 1000er-Quadrat erzeugt und hier durch 1 000 geteilt (Einheitsquadrat), drei Familien: gleichverteilt, geclustert (fünf Gauß-Haufen mit σ = 4 % der Seite) und Gitter mit Rauschen. Eine **greedy-Tour** je Instanz (Start Stadt 0). **Zeigernetz** (12 480 Parameter, eine Schicht aus Attention und Vorwärtsnetz, Dekodierer mit Kontext aus Mittel, Startstadt und aktueller Stadt, Zeigerwerte mit tanh auf 10 begrenzt) und **Merkmalsnetz K** (145 Parameter, sieben Merkmale je Kandidat, ein Netz 7-16-1); beide sind bei n = 20 gleichverteilt mit REINFORCE trainiert (je 960 000 gesampelte Touren, fünf Zufalls-Seeds). Fünf Varianten werden verglichen: **A** (Zeigernetz mit rohen Koordinaten), **A+nt** (dieselben Gewichte, die Eingabe wird beim Einsatz auf das Einheitsquadrat normiert), **An** (die Eingabe ist auch im Training normiert), **B mit s** (Merkmalsnetz, Abstände durch die Skala s der Instanz geteilt) und **B ohne s** (feste Skala s0 = 0,1239, so trainiert); dazu **Aaug** (im Training skaliert und verschoben) und **Amix** (normiert, im Training n = 10, 20, 50) für Experiment 5. Die Formeln stehen im Expander „Mathematische Formulierung“ der App.

## Methodik

* **Wechsel (Skala und Lage):** x → a·x + b mit a = 0,25, a = 10 und b = (3; 2) bei n = 20 und n = 50 in allen drei Familien (18 Zellen). Das Netz sieht die veränderte Instanz, **bewertet wird auf der Basisinstanz** (die Tour ist eine Permutation). Der Nächste Nachbar wählt auf jeder gleichmäßig skalierten und verschobenen Fassung dieselbe Tour; die Netze A+nt, An, B mit s und Amix sind gegen den Wechsel invariant (Konstruktion) und werden darum nicht noch einmal gerechnet (der Anteil gleicher Touren steht in der eingefrorenen Messung und im Test).
* **Größe und Verteilung:** n = 10, 20, 50, 100, 200 in drei Familien, Zufalls-Seed der Instanz 0 bis Zahl der Instanzen − 1 (Zellen siehe „Annahmen“). Referenz bis n = 20 exakt (Held-Karp), ab n = 50 die beste bekannte Lösung (Minimum aus der Guided Local Search von OR-Tools mit kurzem Zeitlimit und allen gebauten Touren); die Lücken sind dort Untergrenzen.
* **Urteil je Zelle:** Unterschied Δ = 100 (L_Netz − L_NN) / L_ref in Prozentpunkten, gepaart über Instanzen (Mittel über die fünf Zufalls-Seeds je Instanz), 95-%-Bootstrap-Intervall über Instanzen (2,5- und 97,5-Prozent-Quantil von 2 000 Ziehungen, SplitMix64). **bricht:** Δ > 0, Intervall ohne 0 und mindestens 4 von 5 Zufalls-Seeds über null; **schlägt:** Δ < 0 und Intervall ohne 0; sonst **hält** (liegt 0 im Intervall: nicht unterscheidbar). Die Schwellen sind gesetzt, nicht abgeleitet.
* **Gegner:** Nächster Nachbar (K = n(n−1)/2) und Nächster Nachbar mit 2-opt (erste Verbesserung bis zum 2-opt-Optimum, nur als Referenzlinie in den Basiszellen; bei n = 200 nur für die ersten 10 Instanzen). Es gibt keine abgestimmten Gegner und keinen Mehrfachstart.
* **Zählung:** K und O nebeneinander (siehe „Training und Zähleinheiten“).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Mittel über fünf Zufalls-Seeds (Gewichts- und Daten-Zufalls-Seed gepaart), Unterschied zum Nächsten Nachbarn in Prozentpunkten der Referenz, 95-%-Bootstrap-Intervall über Instanzen, eine greedy-Tour je Instanz, Instanzen je Zelle 100 (n = 10, 20, 50, 100) und 50 (n = 200) je Familie. Quelle: `nfi_frozen.json`; die Spalte „Test“ nennt den prüfenden Test (Band oder exakt).

| Frage | Ergebnis | Test |
|---|---|---|
| Lücke n = 20 gleichverteilt: Nächster Nachbar, A, A+nt, An, B mit s, B ohne s | 16,48 %; 10,54, 10,59, 10,81; 8,09, 8,28 % | `test_claims::test_luecken_n20` (Bänder) |
| Zellen mit Bruch bei Skala und Lage (18) | A 14; A+nt, An, B mit s je 0; B ohne s 4 | `test_wechselzaehlung` |
| Lücke von A bei Faktor 10, n = 20 / n = 50 gleichverteilt | 136,86 / 281,99 % (Nächster Nachbar 16,48 / 20,90 %) | `test_a_faktor_10` (Bänder) |
| Wechselzellen, in denen die Lücke von A um mindestens 2 Prozentpunkte über der Basiszelle liegt | 18 von 18 | `test_wechselzaehlung` |
| B ohne s bei Faktor 10: Zellen mit Bruch | 4 von 6 (Faktor 0,25 und Verschiebung: 0 von 12) | `test_wechselzaehlung` |
| Basiszellen mit Bruch (15): A, A+nt, An, B mit s, B ohne s | 6; 5; 5; 0; 0 | `test_basiszaehlung` |
| Zellen ab n = 100 mit Bruch (6): A, A+nt, An | 5; 5; 5 | `test_basiszaehlung` |
| Kleinstes n mit Bruch von A, gleichverteilt / geclustert | 100 / 50 | `test_basiszaehlung` |
| B mit s und B ohne s schlagen den Nächsten Nachbarn (15 Basiszellen) | 15 und 15 von 15 | `test_basiszaehlung` |
| Unterschied B mit s, gleichverteilt, n = 100 / n = 200 | −3,04 / −3,74 Prozentpunkte | `test_unterschiede_b` (Bänder) |
| Größter Betrag Lücke(A+nt oder An) minus Lücke(A), Basiszellen | 2,20 Prozentpunkte | `test_normierung_kostet` (Band) |
| Sättigungsanteil der Zeiger-Werte von A (Basis / Faktor 0,25 / Faktor 10 / Verschiebung) | 47,6 / 0,1 / 99,0 / 80,7 % | `test_diagnosen` (Bänder) |
| Gekappter Anteil x1 von B ohne s (Basis / Faktor 10) | 20,7 / 98,5 % | `test_diagnosen` (Bänder) |
| s der Instanz geteilt durch s0 (gleichverteilt, n = 10, 20, 50, 100, 200) | 1,42 / 1,00 / 0,60 / 0,43 / 0,30 | `test_diagnosen` (Band ±0,15) |
| Nächster Nachbar mit 2-opt, n = 50 gleichverteilt (Lücke) | 5,00 % (B ohne s 11,35, B mit s 16,29, A 19,85) | `test_luecken_n50` (Bänder) |
| Teilstichproben (E4): gleiches Urteil wie die Vollstichprobe, 10 Instanzen | B mit s, n = 100 gleichverteilt: 69 von 200; A, n = 200 gleichverteilt: 200 von 200 | `test_teilstichproben` (Bänder) |
| Anteil der Instanzen, auf denen eine gebaute Tour die Referenz unterbietet (n ≥ 50) | höchstens 4 % je Zelle | `test_referenz_unterboten` |
| K je Konstruktion, O (n = 20), Parameter, Handbeispiel | 190; 1 520 / 246 384 / 47 242; 12 480 und 145; Lücke 7,97 % | `test_hand.py`, `test_claims::test_kosten_exakt` (exakt) |

Das Muster der Urteile steht nur in der eingefrorenen Messung und wird dort aus den gespeicherten Werten gezählt (`tests/test_frozen.py`); die Live-Läufe der App (12 Instanzen in Experiment 1) können im Urteil abweichen, die Verdikte nennen Instanzzahl und Zufalls-Seeds.

**Wo sich Planzahlen nicht bestätigt haben** (Vormessung mit `numpy.random`, hier mit SplitMix64 und anderen Instanzen): siehe „Befunde und Korrekturen gegenüber der Vormessung“.

**Zahlen der Literatur** (andere Modelle, nicht vergleichbar, nur zur Einordnung): siehe „Quellen“.

## Training und Zähleinheiten

Die Netze liegen vorgerechnet in `nfi_models/` (je Netz fünf Dateien; Größe, Zufalls-Seeds, Touren und Prüfsumme in `MODELLE.json`). **A** (aus `rechnen-beim-einsatz-demo`, Rezept von `zeigernetz-demo`) und **B mit s** (aus `zugwahl-lernen-demo`) sind Bytekopien (Kopier-Test); **An, B ohne s, Aaug, Amix** sind hier mit `tools/nfi_train_models.py` trainiert (8 000 Schritte × 120 Touren, Lernrate 0,001, Gradientennorm 1, Referenzwert Mittel der Charge beim Zeigernetz; 3 750 Schritte × 256 Touren, Lernrate 0,03, Gradientennorm 5, Referenzwert Mittel der acht Touren je Instanz beim Merkmalsnetz), mit denselben Gewichts- und Daten-Strömen wie A und B mit s (gepaart). Die Länge im Verlust ist immer die auf der Originalinstanz. Die Schleifen sind ohne Veränderung der Eingabe schrittgleich zu den Quellen (Test). **In der App wird nicht trainiert.**
**K** (Kandidatenbewertungen): n(n−1)/2 je Konstruktion für alle drei Verfahren gleich (45, 190, 1 225, 4 950, 19 900 bei n = 10, 20, 50, 100, 200). **O** (grobe Operationen, Multiplikation-Addition = 1, Nächster Nachbar 8 je Kandidat als Annahme): Nächster Nachbar 360, 1 520, 9 800, 39 600, 159 200; Zeigernetz 113 704, 246 384, 764 424, 2 027 824, 6 054 624; Merkmalsnetz (ohne s) 9 298, 47 242, 454 827, 2 837 774, 19 408 225 (mit s zusätzlich n²). Bei n = 20 kostet das Zeigernetz das 162-Fache und das Merkmalsnetz das 31-Fache des Nächsten Nachbarn. **Leserichtung:** K begünstigt die Netze, O benachteiligt sie. Diese Demo fragt „besser bei gleicher Zahl Konstruktionen?“ (eine je Instanz), nicht „bei gleichem Gesamtbudget?“. **Training als dritter Posten, nie in K oder O:** je Netz 960 000 gesampelte Touren, beim Merkmalsnetz 182 400 000 Merkmalszeilen. Zeit ist nie ein Befund.

## Befunde und Korrekturen gegenüber der Vormessung

Die Planzahlen stammen aus einer früheren Messreihe mit eigenen Umsetzungen und `numpy.random` (Instanzen und Netze mit anderem Zufallsstrom). Hier sind die Netze und Instanzen mit SplitMix64 neu gemessen (A und B mit s sind die Netze der Vorgänger). Die Rangfolge der Aussagen bleibt, Zahlen verschieben sich; die Zuordnung „die Normierung ist der Grund für alles“ aus der früheren Fassung gilt nur für Skala und Lage (Größe und Verteilung brechen auch mit Normierung).

| Größe | Planzahl (Vormessung) | Bau (SplitMix64) |
|---|---|---|
| Nächster Nachbar, n = 20 gleichverteilt (Lücke) | 18,10 % | 16,48 % |
| A, A+nt, An, n = 20 gleichverteilt | 10,14 / 10,05 / 9,94 % | 10,54 / 10,59 / 10,81 % |
| B mit s, B ohne s, n = 20 gleichverteilt | 8,99 / 8,56 % | 8,09 / 8,28 % |
| Zellen mit Bruch bei Skala und Lage: A / A+nt, An, B mit s | 14 von 18 / 0 von 18 | 14 von 18 / 0 von 18 |
| Lücke von A bei Faktor 10, n = 20 / n = 50 gleichverteilt | 134,01 / 288,31 % | 136,86 / 281,99 % |
| B ohne s bei Faktor 10: Zellen mit Bruch | 2 von 6 | **4 von 6** (breitere Instanzmenge: 100 statt 30 und 10 Instanzen) |
| Basiszellen mit Bruch: A / A+nt, An | 6 / 5, 5 von 15 | 6 / 5, 5 von 15 |
| Kleinstes n mit Bruch von A: gleichverteilt / geclustert | 100 / 50 | 100 / 50 |
| B mit s schlägt (15 Basiszellen) / B ohne s | 15 / 14 (n = 200 geclustert „nicht unterscheidbar“) | 15 / **15** (n = 200 geclustert −2,98 [−4,28; −1,56] mit 50 statt 10 Instanzen) |
| Unterschied B mit s, gleichverteilt, n = 100 / n = 200 | −5,51 / −4,47 | **−3,04 / −3,74** |
| Größter Betrag Lücke(A+nt oder An) minus Lücke(A), Basiszellen | 3,74 Prozentpunkte | 2,20 Prozentpunkte |
| Sättigungsanteil von A (Basis / 0,25 / 10 / Verschiebung) | 49,0 / 0,2 / 99,1 / 85,2 % | 47,6 / 0,1 / 99,0 / 80,7 % |
| Gekappter Anteil x1 von B ohne s (Basis / Faktor 10) | 20,0 / 98,3 % | 20,7 / 98,5 % |
| s geteilt durch s0 (n = 10, 20, 50, 100, 200) | 1,50 / 0,98 / 0,62 / 0,43 / 0,29 | 1,42 / 1,00 / 0,60 / 0,43 / 0,30 |
| Nächster Nachbar mit 2-opt, n = 50 gleichverteilt | 5,6 % (frühere Messreihe, 200 Instanzen) | 5,00 % (100 Instanzen) |

Was sich geändert hat: (1) **B ohne s bricht bei Faktor 10 häufiger als geplant** (4 von 6 Zellen statt 2 von 6; die Zellen mit „hält“ sind n = 20 Gitter, 2 von 5 Zufalls-Seeds, und n = 50 geclustert). Die Aussage „B ohne s bricht nur teilweise“ steht damit auf 4 von 6, mit starker Streuung über die Zufalls-Seeds. (2) **B mit s und B ohne s schlagen den Nächsten Nachbarn in allen 15 Basiszellen**; der Vorsprung bei n = 100 und n = 200 gleichverteilt ist kleiner als geplant (B mit s −3,04 und −3,74 statt −5,51 und −4,47). (3) Die Zahlen der Vormessung für den Zeigernetz-Bruch bei Skala und Lage (14 von 18) und für die Basiszellen (6, 5, 5 von 15) bestätigen sich. (4) **Die Annahme „das Zeigernetz bricht bei n ≥ 100 fast überall“ ist genauer:** 5 von 6 Zellen (n = 100 Gitter hält, −0,47 [−1,67; +0,75]); bei n = 50 bricht nur geclustert, und mit Normierung nur als „nicht unterscheidbar“. Die Zählregel folgt `znz_cost.py` (Dekodierer 35 184 statt 34 160 der Vormessung, Nächster Nachbar 8 je Kandidat statt n²: bei n = 20 der Faktor 162 statt 613 für das Zeigernetz, 31 statt 119 für das Merkmalsnetz).

Abweichungen vom Plan im Bau: (a) die feste Skala s0 ist 0,1239 (Mittel von s über 20 000 Trainingsinstanzen, Strom `ZWECK_TRAIN_X`; die Vormessung hatte 0,1236 aus 1 000 Instanzen); (b) B ohne s und B mit s benutzen dieselben Ströme (Gewichte, Trainingsinstanzen, Startstädte, Ziehungen von `zugwahl-lernen-demo`), die Zwecke 51 und 52 des Plans sind nicht belegt; (c) die Referenz ab n = 50 ist die Guided Local Search von OR-Tools mit 2, 4 und 6 Sekunden je Instanz bei n = 50, 100 und 200 (kurz gehalten wegen der Rechenlast), kombiniert mit allen gebauten Touren; die Lücken ab n = 50 sind Untergrenzen, auf höchstens 4 % der Instanzen je Zelle unterbietet eine gebaute Tour die Metaheuristik; (d) die Lernrate des Merkmalsnetzes (0,03) ist nicht neu abgestimmt (der Plan sah ein Raster {0,01; 0,03; 0,1} vor); (e) A und B mit s sind Bytekopien der Netze aus den Vorgängern, nur An, B ohne s und die Netze des Mischtrainings sind hier trainiert; (f) die Kosten des Merkmalsnetzes stehen ohne s (47 242) und mit s (47 642, zusätzlich n² für die Skala, wie in `rechnen-beim-einsatz-demo`).

Experiment 5 (im Plan bedingt) ist gemessen: Aaug und Amix je 8 000 Schritte × 120 Touren, fünf Zufalls-Seeds (Aaug 145 bis 153 s, Amix 258 bis 270 s je Lauf auf dem Entwicklungsrechner). Die Schwellen für den Satz zur Ursache standen vor der Messung fest (`nfi_constants`); weil Aaug nicht lernte, kann „am Training“ nicht belegt werden (`ursache_satz`), und Amix bricht ab n = 100 in 5 von 6 Zellen, was die Schwelle für „spricht gegen nur das Training bei einer Größe“ erfüllt.

**Ausblick (frühere Messreihe, nicht neu gemessen, Messung dieser README, nicht im Test):** ein bei n = 50 direkt trainiertes Zeigernetz erreichte dort (200 Instanzen, andere Instanzen und anderes Training als hier) 19,72 % Lücke gegen 20,82 % für das bei n = 20 trainierte, ohne nachweisbaren Unterschied; der Nächste Nachbar lag bei 22,3 %. Die App nennt das als Textzeile bei Experiment 2, nicht in der Tabelle.

**Offen und nicht gemessen:** Linux-Lauf der Tests (Docker nicht gestartet); Streamlit-Cloud-Laufzeit und -Speicher; dunkles Design und Telefonbreite; Auswahl per Mausklick in Plotly; der unabhängige Aussagenprüfer steht vor dem ersten Push aus.


## Ehrliche Grenzen

**Annahmen**
* Zwei eigene, kleine Netze von Hand in numpy (12 480 und 145 Parameter, eine Schicht beim Zeigernetz, kein Kodierer beim Merkmalsnetz), Training **nur bei n = 20 gleichverteilt**, je 960 000 gesampelte Touren, fünf Zufalls-Seeds je Netz. Kein Kool- oder POMO-Modell, keine Aussage über diese Modelle.
* **Eine greedy-Tour je Instanz**, Start Stadt 0; Mehrfachstart, Sampling und Beam sind für das Zeigernetz nicht gemessen. Gleiche Zahl Konstruktionen (eine je Instanz), aber das Netz kostet in O ein Vielfaches des Nächsten Nachbarn (Zählregel oben).
* Alle Instanzen sind erzeugt (drei Familien: gleichverteilt, Gruppen, Gitter), keine realen Instanzen, keine anderen Fahrzeuge. Referenz bis n = 20 exakt, ab n = 50 die beste bekannte Lösung (Metaheuristik und alle gebauten Touren): die Lücken sind dort Untergrenzen, die gepaarten Unterschiede dagegen kaum betroffen (die zu große Referenz staucht sie nur leicht).

**Grenzen**
* **Latte Nächster Nachbar:** „hält“ ist kein Lob. Nächster Nachbar mit 2-opt liegt weit vor beiden Netzen (Experiment zur Größe); die Netze schlagen nur den Nächsten Nachbarn.
* Drei Wechselwerte (Faktor 0,25 und 10, Verschiebung (3; 2)), nur gleichmäßige Skalierung, keine Drehung, keine Achsen mit verschiedenem Maßstab; Wechsel nur bei n = 20 und n = 50.
* Instanzen je Zelle: 100 bei n = 10, 100 bei n = 20, 100 bei n = 50, 100 bei n = 100, 50 bei n = 200 (je Familie); bei n = 100 und n = 200 sind die Intervalle breit.
* Ursachen sind **nicht untersucht**: warum das Zeigernetz bei großem n bricht, warum das Merkmalsnetz ohne s bei n = 50 und 100 gleichverteilt besser ist als mit s. Sättigung, Kappung und s/s0 sind Hinweise. Unterschiede zwischen den beiden Netzen gelten für diese beiden Netze (Eingabe und Bauweise unterscheiden sich), nicht für „Attention“ als Ursache.
* Mischtraining: nur so weit Experiment 5 gemessen ist.

**Bekannte Probleme und Unsicherheiten**
* Die Netze liegen vorgerechnet im Repo; in der App wird nicht trainiert. Laufzeit und Speicher auf Streamlit Cloud sind nicht gemessen.
* Die Zufalls-Seeds der Netze streuen stark (zum Beispiel beim Merkmalsnetz ohne s bei Faktor 10); die Urteile „bricht“ und „schlägt“ stehen deshalb nur mit Zahl der Instanzen und der Zufalls-Seeds.
* Die Lernrate des Merkmalsnetzes (0,03) ist von Stück 4 übernommen und für beide Skalen nicht neu abgestimmt; die Lernrate des Zeigernetzes (0,001) ist nur für n = 20 abgestimmt.
* Plotly-Auswahl per Mausklick ist nicht getestet; dunkles Design und Telefonbreite nicht geprüft.

## Verifikation

**233 Tests in 18 Dateien** (ein Prozess, rund 65 bis 80 s je nach Last, ohne xdist; größte Einzelläufe: Einzelknopf bei n = 200 am Reglermaximum und die Presets bei n = 100 und n = 200). Zusammensetzung:

* **Gradienten-Check zuerst und dauerhaft:** Zeigernetz (d = 8, n = 6, normierte Eingabe) und Merkmalsnetz (Politikverlust bei fester Skala) gegen zentrale endliche Differenzen, relative Abweichung unter 1e-6 (`tests/test_oracle_netz_fremde_instanzen.py`).
* **Orakel** (anderer Rechenweg): Held-Karp gegen Aufzählung (n = 5 bis 8, je 20 Instanzen) und gegen CP-SAT (n = 12); die Referenz der Guided Local Search gegen Held-Karp bei n = 12; die sieben Merkmale in Vektorfassung gegen die Schleifenfassung (20 Instanzen, drei Schritte); der Vorwärtspass des Zeigernetzes gegen eine Fassung in reinem Python mit `math` (n = 5, d = 4); Zählung der Merkmalszeilen und Vergleiche (n(n−1)/2); Bootstrap-Intervall gegen das t-Intervall; **Testmenge und Trainingsmenge getrennt** (keine gemeinsamen Orte); Vektorziehung gleich 50 Einzelziehungen des skalaren SplitMix64.
* **Handbeispiel von Hand** (fünf Orte, `tests/test_hand.py`, dieselben Toleranzen wie `neuro-kombinatorik-planung/handbeispiele_05.py`): Nächster Nachbar, Optimum durch Aufzählung (Lücke 7,97 %), Skala und Lage, Spielzeug-Zeigerstufe, sieben Merkmale mit und ohne s, Parameterzahlen, gezählte Matrixprodukte gegen die Formeln, K und O, die Urteilsregel an zehn Berichtszellen.
* **Kopier-Tests** (`tests/test_copies.py`): alle Kopien und Modelle gleichen ihren Quellen (übersprungen ohne Quellordner), die Konstantendateien führen dieselben Namen und Werte, kein `numpy.random` im Kern.
* **Eigenschaftstests:** die Netze A+nt, An, B mit s und Amix liefern unter allen Wechseln dieselbe Tour; A mit roher Eingabe ändert die Tour bei Faktor 10 (Zweig-Test); B ohne s ist gegen Verschieben invariant und gegen Faktor 10 nicht; Länge unter Wechsel gleich Faktor mal Länge; jede Tour ist eine Permutation.
* **Trainingsschleifen:** ohne Veränderung der Eingabe schrittgleich zu den Quellen, jede Variante weicht von der Nachbarvariante ab.
* **Eingefrorene Messung** (`tests/test_frozen.py`): Schema, endliche Werte, Instanzzahlen, Referenz höchstens gleich jeder gespeicherten Länge, Referenzen bis n = 20 exakt, **Neu-Rechnen von drei Instanzen je Zelle (n = 20 und 50, Basis und drei Wechsel) als Band** (höchstens 0,5 Prozent im Zellenmittel der Länge, nie bytegleich).
* **Verdikte und Claims** (`tests/test_verdicts.py`, `tests/test_claims.py`): Schwellen und jede Zahl der Texte an handgesetzten Zeilen; jede Zahl der Tabelle oben als Band oder exakt; je Preset die erwartete Verdikt-Zeile.
* **Oberfläche:** AppTest (jedes Preset, jeder Regler am Rand, Handbeispiel ohne Netzläufe), Layout (Schnellstart in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, jede Änderung eines Reglers hebt die Markierung auf), Worst-Case am Reglermaximum, Footer, Laufzeit-Importe (`tests/test_runtime_imports.py`), `modul_frisch`.
* **Browser** (Port 10134, Streamlit 1.65): Primäransicht, Experimente, Lupe angesehen; die Konsole zeigte beim ersten Laden im vorderen Tab vier Plotly-Meldungen „<text> attribute y: Expected length, -Infinity“, im Hintergrund-Tab (n = 50 geclustert) keine; nach dem Zeichnen hatte keines der zwölf Diagramme ein `text`-Element mit nicht endlichem y, kein Ausnahme-Element. Die Ursache ist nicht geklärt. Nicht geprüft: dunkles Design, Telefonbreite, Mausklick-Auswahl in Plotly.
* **Wächter:** ruff (E, F, I), Text-Lint, Links, Testzahl der README, Requirements-Prüfung, Muster-Suche; `claim_review scan` und eigene Durchsicht der Sätze mit Absolut- und Urteilswörtern. **Der unabhängige Aussagenprüfer steht vor dem ersten Push aus; der Linux-Lauf der Tests (Docker) ist nicht gelaufen.**

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (Aufbau nach DEMO-PLAYBOOK §2): Kopf, Einleitung, Erklär-Expander, Primäransicht, fünf Experimente, Kosten, Grenzen, Formeln |
| `nfi_constants.py`, `nfik_constants.py` | Regler-Grenzen, Wechselwerte, Zwecke der Zufalls-Seeds, Marken der Urteile, Farben, Presets; die zweite Datei führt die Namen der Kopien aus der Zugwahl-Demo |
| `nfi_rng.py`, `nfi_rngblock.py`, `nfi_rngvec.py` | **Kopien** (Zufall: SplitMix64, Block- und Vektorziehung, bitgleich zum skalaren Strom) |
| `nfi_scenario.py`, `nfi_instances.py` | **Kopien** (Instanzen der drei Familien, Normierung der Eingabe) |
| `nfi_net.py`, `nfi_train.py` | **Kopien** (Zeigernetz von Hand in numpy: Kodierer, Dekodierer, Rückwärtspass; Adam, REINFORCE) |
| `nfi_exact.py`, `nfi_gegner.py`, `nfi_stats.py`, `nfi_cost.py` | **Kopien** (Held-Karp; Nächster Nachbar und 2-opt mit Zähler; Bootstrap; Zählregel K und O) |
| `nfik_mlp.py`, `nfik_featnet.py`, `nfik_policy.py` | **Kopien** (Netz 7-16-1, Merkmalsnetz K mit Rollout, Politikgradient) |
| `modul_frisch.py` | Kopie: lädt geänderte Hilfsmodule nach einem Update neu |
| `nfi_transform.py` | Wechsel (Skalieren, Verschieben), Testmengen aller Familien, Nächster Nachbar und Tourlängen für viele Instanzen |
| `nfi_models.py`, `nfi_models/` | Laden der vorgerechneten Netze (A, An, Aaug, Amix: Zeigernetz; Bmit, Bohne: Merkmalsnetz), `MODELLE.json` mit Größe und Prüfsumme |
| `nfi_trainloops.py` | Trainingsschleifen der Varianten (nur Werkzeuge und Tests) |
| `nfi_eval.py` | Touren der Netze, Längen auf der Basisinstanz, Zellenstatistik |
| `nfi_diagnose.py` | Sättigung der Zeiger-Werte, Kappung der Merkmale, s/s0 (Hinweise, keine Ursachenmessung) |
| `nfi_kosten.py` | K und O je Konstruktion, Rechengrenze je Klick |
| `nfi_evaluation.py` | Urteilsregel `klasse_zelle`, Zahlenformate und alle Verdikte als reine Funktionen |
| `nfi_experiments.py`, `nfi_befunde.py` | Lesen der eingefrorenen Messung, Live-Zellen, Teilstichproben; Kennzahlen für README und Tests |
| `nfi_hand.py` | Handbeispiel (fünf Orte) in Schleifen |
| `nfi_presets.py`, `nfi_layout.py`, `nfi_visualization.py`, `nfi_texts.py` | Presets und Permalink, fester Kopf, Plotly-Figuren, Texte |
| `nfi_frozen.json`, `nfi_refs.json` | eingefrorene Messung und Referenzen |
| `tools/` | `nfi_kopieren.py`, `nfi_train_models.py`, `nfi_make_refs.py`, `nfi_make_frozen.py`, `nfi_modelle_info.py`, `nfi_zahlen.py` (Zwischenstände in `tools/_zwischen/`, nicht im Repo) |
| `tests/` | siehe „Tests“ |
| `requirements.txt` | Laufzeit: streamlit, numpy, plotly (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest und ortools (Referenzen und ein Orakel-Test) |

## Bewusst nicht enthalten

* **Training in der App.** Die Netze sind vorgerechnet (`nfi_models/`), die Skripte stehen in `tools/`; die Demo prüft Netze im Einsatz auf fremden Instanzen, sie trainiert nicht.
* **Mehrfachstart, Sampling, Strahlsuche, aktive Suche** (gehören zu `rechnen-beim-einsatz-demo`), **Netz als Startlösung**, **abgestimmte Gegner und Rangtest** über mehrere Gegner (gehören zu `netz-gegen-gegner-demo`), **Referenzwert-Formen** (`basislinie-lernen-demo`), **Zugwahl** (`zugwahl-lernen-demo`). Hier gibt es **eine** greedy-Tour je Instanz.
* **n = 500** und größer, **reale Instanzen** (TSPLIB), **Drehung und Achsenstreckung**, mehr als drei Wechselwerte, Nebenbedingungen (Kapazität, Zeitfenster), Heatmap-Ansätze, Sprachmodelle, ein Netz nur mit den drei skalenfreien Merkmalen (x2, x6, x7).
* **Ursachen der Brüche** (warum das Zeigernetz bei großem n bricht, warum das Merkmalsnetz ohne s bei n = 50 und 100 gleichverteilt besser ist als mit s): nicht untersucht.
* **Nutzen im Betrieb**: nicht belegt; alle Instanzen sind erzeugt.

## Quellen (mit Belegstufe)

* Vinyals, Fortunato, Jaitly 2015, „Pointer Networks“ (Belegstufe V): Zeigernetz.
* Joshi, Laurent, Bresson 2019, „An Efficient Graph Convolutional Network Technique for the Travelling Salesman Problem“, arXiv 1906.01227 (Belegstufe V, Tabelle 2): ein Attention-Modell (Kool), bei n = 20 trainiert, hat bei n = 20 0,08 %, bei n = 50 1,78 % und bei n = 100 22,61 % Lücke (1 280 Samples); anderes Modell, nicht mit den Zahlen dieser Demo vergleichbar.
* Joshi, Cappart, Rousseau, Laurent 2022, „Learning the Travelling Salesperson Problem Requires Rethinking Generalization“, Constraints 27 (Belegstufe V für den Inhalt): ein nur bei TSP20 trainiertes Modell verallgemeinert nicht auf große Größen (Training bei TSP20 bis TSP50 hilft bei TSP200 dagegen); ein nicht gelerntes Einfügeverfahren ist bei TSP200 im Budget schwer zu schlagen. Aussagen über deren Modelle, hier nicht nachgemessen.
* Luo u. a. 2023, LEHD (Belegstufe V, Tabelle 1): POMO mit achtfacher Abbildung, bei kleinen Größen trainiert, liegt bei TSP100, 200, 500 und 1 000 bei 0,134 %, 1,533 %, 22,187 % und 40,570 % Lücke; anderes Modell.
* Liu, Zhang, Tang, Yao, „How Good Is Neural Combinatorial Optimization? A Systematic Evaluation on the TSP“, arXiv 2209.10913 (Belegstufe V, Abschnitt IV-B und IV-D): ein auf gleichverteilten Städten trainiertes Netz ist auf geclusterten Instanzen schlechter, und Mischtraining erreicht die beste Einzelverteilung nicht.
* Survey von Ba u. a., arXiv 2602.21761 (Belegstufe V für Teile, Abschnitt VI-D): vermutet vor allem die enge Trainingsverteilung als Ursache der Schwäche (nicht eine grundsätzliche Grenze der Netze), nennt daneben auch einfache algorithmische Rahmen.
* Nicht gelesen und nicht zitiert: Arbeiten zu Augmentierung und Mischtraining im Einzelnen.

## Verweise

Live-Demos: [`pretrained-net-demo`](https://sebastianhanisch-pretrained-net-demo.streamlit.app/) (Zeitreihen, Zero-Shot und Feintuning auf neuen Depots), [`mappo-demo`](https://sebastianhanisch-mappo-demo.streamlit.app/) (Held-out-Instanzen bei Netzpolitik), [`vergleich-kippt-demo`](https://sebastianhanisch-vergleich-kippt-demo.streamlit.app/) (die Mahnung zu Experiment 4), [`einstellen-lohnt-demo`](https://sebastianhanisch-einstellen-lohnt-demo.streamlit.app/) (abgestimmte Gegner), [`policy-gradient-demo`](https://sebastianhanisch-policy-gradient-demo.streamlit.app/), [`actor-critic-demo`](https://sebastianhanisch-actor-critic-demo.streamlit.app/), [`attention-transformer-demo`](https://sebastianhanisch-attention-transformer-demo.streamlit.app/) und [`graph-transformer-demo`](https://sebastianhanisch-graph-transformer-demo.streamlit.app/).
Nur mit Namen (kein Live-Link in deren README): `regel-verteilungswechsel-demo` (ähnliche Frage, anderes Fahrzeug: Einplanung und Packen mit erzeugten Regeln), `meta-ebene-kosten-demo`, `hyperheuristik-demo`, `zeigernetz-demo` und `zugwahl-lernen-demo` (Quellen der Kopien), `rechnen-beim-einsatz-demo` (Quelle der Zeigernetz-Modelle), `basislinie-lernen-demo`, `netz-gegen-gegner-demo`.
`nfi_frozen.json` ist die Quelle der Zahlen für `netz-gegen-gegner-demo` (Basiszellen, Netz A und Merkmalsnetz B bei n = 20 und 50, Nächster Nachbar).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Einmalig neu erzeugen (Werkzeuge, nicht für die App): `python tools/nfi_kopieren.py` (Kopien und Modelle aus den Quellordnern), `python tools/nfi_train_models.py --netze Bohne An Aaug Amix` (Netze trainieren, setzt fort), `python tools/nfi_make_refs.py` (Referenzen), `python tools/nfi_make_frozen.py --stage base|wechsel|diagnose|final` (eingefrorene Messung), `python tools/nfi_zahlen.py` (Kennzahlen für diese README).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und numpy.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Neuronale kombinatorische Optimierung](https://sebastianhanisch.net/konzepte-neuronale-kombinatorische-optimierung.html).

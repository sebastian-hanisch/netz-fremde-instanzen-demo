"""Texte der App: Titel, Einleitung, Erklär-Expander, Annahmen und Grenzen (= README-Abschnitt, Test), Formeln.

Zahlen, die aus der Messung stammen, stehen nicht hier, sondern werden in der App aus `nfi_frozen.json` gelesen (Verdikte, Zählzeilen); hier stehen nur Zahlen der Konstruktion
(Parameterzahlen, K und O nach der Zählregel) und die Größe der Messung aus `nfi_constants`."""

import nfi_constants as C
import nfi_evaluation as VD
import nfi_kosten as KO

TITEL = "🧭 Das Netz auf anderen Rundtouren"
UNTERTITEL = "Zeigernetz und Merkmalsnetz bei Skala, Lage, Größe und Verteilung"

_K20 = KO.kosten_je_konstruktion(20)

INTRO = """Ein Netz, das bei **20 Orten gleichverteilt** trainiert wurde, bekommt Rundtouren, die anders aussehen als im Training: kleiner oder größer gezeichnet, verschoben, mit mehr Orten, in Gruppen oder auf einem Gitter.
Die Demo trennt zwei Fragen: **Skala und Lage** (derselbe Plan, nur anders gemessen, mit einer Normierung der Eingabe als Schalter) und **Größe und Verteilung** (wirklich andere Instanzen).
Zwei kleine Netze treten mit fünf Zufalls-Seeds des Trainings gegen den **Nächsten Nachbarn** an: ein **Zeigernetz (Pointer Network)** mit Koordinaten als Eingabe (Zeiger = Gewicht, mit dem das Netz auf den nächsten Ort zeigt)
und ein **Merkmalsnetz** mit sieben Merkmalen aus Abständen. Die Latte ist niedrig, und alle Instanzen sind erzeugt. In dieser Reihe prüft diese Demo Netze außerhalb ihrer Trainingsverteilung systematisch; die Vorgänger (`rechnen-beim-einsatz-demo`, `zeigernetz-demo`, `zugwahl-lernen-demo`) zeigen das nur als Ausblick bei n = 50.
Unten stehen die Experimente zu Skala und Lage, Größe, Verteilung, Stichprobe und Mischtraining, `🚧 Wo die Annahmen enden` und die Formeln."""

EINFACHE_WORTE = """Ein Netz lernt an Beispielen, und was es lernt, hängt davon ab, was es sieht. Zeigt man ihm denselben Plan in einem anderen Maßstab oder an anderer Stelle, kann es, wenn es rohe Koordinaten bekommt,
völlig falsch entscheiden, obwohl sich an der Aufgabe nichts geändert hat. Das lässt sich mit einem Handgriff beheben: die Koordinaten vorher in ein festes Quadrat rechnen.
Ganz anders ist es, wenn es mehr Orte gibt oder die Orte in Gruppen liegen. Dann sieht das Netz wirklich etwas Neues, und der Handgriff hilft nicht. Das Netz, das nur Abstände und Rangfolgen sieht,
kommt dort in dieser Messung besser durch als das, das Koordinaten sieht; ob das an der Eingabe oder an der Bauweise liegt, ist damit nicht gesagt. Und auch das einfache Abgehen zum nächsten Ort ist keine hohe Latte."""

SO_SIEHT_EIN_NETZ = """Was jedes Verfahren von einer Instanz sieht, und was sich bei einem Wechsel ändert:

| Verfahren | Eingabe | Skala und Lage ändern | Mehr Orte, andere Verteilung |
|---|---|---|---|
| Nächster Nachbar | Abstände | nichts: dieselbe Tour | dasselbe Verfahren, andere Instanz |
| Zeigernetz, rohe Eingabe (A) | Koordinaten | die Eingabe ändert sich | andere Dichte, mehr Kandidaten |
| Zeigernetz, normiert (A+nt, An) | Koordinaten im Einheitsquadrat | nichts: gleiche Eingabe (Konstruktion) | andere Dichte, mehr Kandidaten |
| Merkmalsnetz mit s (B mit s) | Abstände geteilt durch die Skala s der Instanz | nichts: gleiche Merkmale (Konstruktion) | andere Dichte, mehr Kandidaten |
| Merkmalsnetz mit fester Skala (B ohne s) | Abstände geteilt durch die feste Skala s0 | Faktor ändert die Merkmale, Verschiebung nicht | wie oben |

Die Netze sind bei 20 Orten gleichverteilt im Einheitsquadrat trainiert (je 960 000 gesampelte Touren). Alle Zahlen der Experimente gelten für eine greedy-Tour je Instanz."""

ERKLAERTABELLE = f"""| Name | Eingabe | Was es tut | Was es hier kostet |
|---|---|---|---|
| Zeigernetz (Pointer Network) | Koordinaten der Orte | Kodierer macht aus den Koordinaten Vektoren, der Dekodierer „zeigt“ in jedem Schritt mit einem Gewicht je freiem Ort auf den nächsten (Softmax über die freien Orte) | {VD.zahl(C.ZEIGER_PARAMS)} Parameter; {VD.zahl(_K20['O_zeiger'])} grobe Operationen je Tour bei n = 20 |
| Merkmalsnetz | sieben Merkmale je Kandidat (Abstand, Rang, Dichte, Anteil besucht, Verhältnis zum nächsten) | ein kleines Netz 7-16-1 bewertet jeden freien Ort; die Tour entsteht schrittweise | {C.MERKMAL_PARAMS} Parameter; {VD.zahl(_K20['O_merkmal'])} grobe Operationen je Tour bei n = 20 |
| Eingabenormierung | Koordinaten | (x − Minimum je Achse) geteilt durch die größte Spannweite: unabhängig von Verschiebung und Skalierung | eine Zeile Rechnung |
| Nächster Nachbar | Abstände | geht immer zum nächsten freien Ort | K = n(n−1)/2 Vergleiche, 8 grobe Operationen je Vergleich (Annahme) |
| Bootstrap-Intervall | Unterschiede je Instanz | zieht Instanzen mit Zurücklegen und gibt das 2,5- und 97,5-Prozent-Quantil der Mittel an | keine Kosten in der Tour |"""

ZAEHLREGEL = f"""Gezählt wird nebeneinander in **K** (Kandidatenbewertungen) und **O** (grobe Operationen), das Training steht als dritter Posten daneben.

* **K**: eine Konstruktion kostet n(n−1)/2 Kandidatenbewertungen, für den Nächsten Nachbarn, das Zeigernetz und das Merkmalsnetz gleich ({VD.zahl(KO.k_konstruktion(20))} bei n = 20, {VD.zahl(KO.k_konstruktion(200))} bei n = 200). K begünstigt die Netze: ein Zeigerschritt oder eine Merkmalszeile kostet mehr als ein Vergleich.
* **O**: Multiplikation und Addition zählen 1; das Zeigernetz aus den Formen der Matrixprodukte einzeln gezählt, der Nächste Nachbar 8 je Kandidat (Annahme, nur auf Zehnerpotenzen belastbar), das Merkmalsnetz mit der Merkmalsbildung. O benachteiligt die Netze. Bei n = 20: Nächster Nachbar {VD.zahl(_K20['O_nn'])}, Zeigernetz {VD.zahl(_K20['O_zeiger'])} (das {VD.faktor(_K20['f_zeiger'])}-Fache), Merkmalsnetz {VD.zahl(_K20['O_merkmal'])} (das {VD.faktor(_K20['f_merkmal'])}-Fache; mit der Skala s der Instanz {VD.zahl(_K20['O_merkmal_mit_s'])}).
* **Training (nie in K oder O)**: Zeigernetz {VD.zahl(C.TOUREN_ZEIGER)} gesampelte Touren ({VD.zahl(C.SCHRITTE_ZEIGER)} Schritte × {C.BATCH} Touren), Merkmalsnetz {VD.zahl(C.TOUREN_MERKMAL)} Touren und {VD.zahl(C.ZEILEN_MERKMAL)} Merkmalszeilen.
* Hier gilt die Frage „besser bei gleicher Zahl Konstruktionen?“ (eine je Instanz). Die Frage „besser bei gleichem Gesamtbudget?“ stellt diese Demo nicht. Zeit ist nie ein Befund."""

# ---------------------------------------------------------------------------------------------------------------- Überschriften der Experimente
E0_TITEL = "Eine Instanz, drei Touren"
E1_TITEL = "🔬 Bricht das Netz, wenn dieselbe Instanz kleiner, größer oder verschoben gezeichnet wird?"
E2_TITEL = "🔬 Bricht das Netz bei mehr Orten?"
E3_TITEL = "🔬 Und bei anderer Verteilung der Orte?"
E4_TITEL = "🔬 Wie viele Instanzen braucht eine solche Aussage?"
E5_TITEL = "🔬 Rettet Training mit gemischten Daten, was die Normierung nicht rettet?"

E1_NUTZT = "Benutzt: Familie, Städtezahl, Zufalls-Seed, Skalenfaktor, Verschiebung, Normierung, Skala des Merkmalsnetzes, Zahl der Seeds (Seitenleiste); hier Zahl der Instanzen."
E2_NUTZT = "Benutzt: Familie, Normierung und Skala des Merkmalsnetzes (Seitenleiste). Die Kurve und die Tabelle sind eingefroren, nur der Knopf rechnet live."
E3_NUTZT = "Benutzt: Städtezahl (Seitenleiste), eingefrorene Messung."
E4_NUTZT = "Benutzt: nichts aus der Seitenleiste; Zelle, Netz, Teilstichprobe und Wiederholungen stehen hier. Rechnet live aus den eingefrorenen Einzelwerten."
E5_NUTZT = "Benutzt: Familie (Seitenleiste), eingefrorene Messung."

LUPE_TEXT = """**Lupe: Sättigung bei Faktor 10 (ein Hinweis auf den Mechanismus).** Der Zeiger bewertet einen freien Ort mit u = C · tanh(⟨Kontext, Schlüssel⟩ / √d) und C = 10. Wächst die Eingabe um den Faktor 10, wachsen die Skalarprodukte,
tanh läuft gegen 1 und die Werte der freien Orte liegen in vielen Schritten am Rand: das Netz kann sie dann kaum noch unterscheiden. Das Spielzeug unten rechnet nur diese Form (kein trainiertes Netz) an den fünf Orten des Handbeispiels;
die Anteile darunter sind Diagnosen am trainierten Netz, **Hinweise, keine Ursachenmessung**."""

# ---------------------------------------------------------------------------------------------------------------- Annahmen, Grenzen, bekannte Probleme (gleich README)
ASSUMPTIONS_TITEL = "⚠️ Annahmen, Grenzen und bekannte Probleme"


def assumptions_markdown() -> str:
    """Text von App und README (der Test sucht ihn in der README)."""
    inst = C.INSTANZEN
    return f"""**Annahmen**
* Zwei eigene, kleine Netze von Hand in numpy ({VD.zahl(C.ZEIGER_PARAMS)} und {C.MERKMAL_PARAMS} Parameter, eine Schicht beim Zeigernetz, kein Kodierer beim Merkmalsnetz), Training **nur bei n = 20 gleichverteilt**, je {VD.zahl(C.TOUREN_ZEIGER)} gesampelte Touren, fünf Zufalls-Seeds je Netz. Kein Kool- oder POMO-Modell, keine Aussage über diese Modelle.
* **Eine greedy-Tour je Instanz**, Start Stadt 0; Mehrfachstart, Sampling und Beam sind für das Zeigernetz nicht gemessen. Gleiche Zahl Konstruktionen (eine je Instanz), aber das Netz kostet in O ein Vielfaches des Nächsten Nachbarn (Zählregel oben).
* Alle Instanzen sind erzeugt (drei Familien: gleichverteilt, Gruppen, Gitter), keine realen Instanzen, keine anderen Fahrzeuge. Referenz bis n = 20 exakt, ab n = 50 die beste bekannte Lösung (Metaheuristik und alle gebauten Touren): die Lücken sind dort Untergrenzen, die gepaarten Unterschiede dagegen kaum betroffen (die zu große Referenz staucht sie nur leicht).

**Grenzen**
* **Latte Nächster Nachbar:** „hält“ ist kein Lob. Nächster Nachbar mit 2-opt liegt weit vor beiden Netzen (Experiment zur Größe); die Netze schlagen nur den Nächsten Nachbarn.
* Drei Wechselwerte (Faktor 0,25 und 10, Verschiebung (3; 2)), nur gleichmäßige Skalierung, keine Drehung, keine Achsen mit verschiedenem Maßstab; Wechsel nur bei n = 20 und n = 50.
* Instanzen je Zelle: {inst[10]} bei n = 10, {inst[20]} bei n = 20, {inst[50]} bei n = 50, {inst[100]} bei n = 100, {inst[200]} bei n = 200 (je Familie); bei n = 100 und n = 200 sind die Intervalle breit.
* Ursachen sind **nicht untersucht**: warum das Zeigernetz bei großem n bricht, warum das Merkmalsnetz ohne s bei n = 50 und 100 gleichverteilt besser ist als mit s. Sättigung, Kappung und s/s0 sind Hinweise. Unterschiede zwischen den beiden Netzen gelten für diese beiden Netze (Eingabe und Bauweise unterscheiden sich), nicht für „Attention“ als Ursache.
* Mischtraining: nur so weit Experiment 5 gemessen ist.

**Bekannte Probleme und Unsicherheiten**
* Die Netze liegen vorgerechnet im Repo; in der App wird nicht trainiert. Laufzeit und Speicher auf Streamlit Cloud sind nicht gemessen.
* Die Zufalls-Seeds der Netze streuen stark (zum Beispiel beim Merkmalsnetz ohne s bei Faktor 10); die Urteile „bricht“ und „schlägt“ stehen deshalb nur mit Zahl der Instanzen und der Zufalls-Seeds.
* Die Lernrate des Merkmalsnetzes (0,03) ist von Stück 4 übernommen und für beide Skalen nicht neu abgestimmt; die Lernrate des Zeigernetzes (0,001) ist nur für n = 20 abgestimmt.
* Plotly-Auswahl per Mausklick ist nicht getestet; dunkles Design und Telefonbreite nicht geprüft."""


ASSUMPTIONS_MARKDOWN = assumptions_markdown()

# ---------------------------------------------------------------------------------------------------------------- Formeln
FORMEL_LESEHILFE = """**Lesehilfe.** *Gegeben:* eine Instanz (Orte in der Ebene), zwei trainierte Netze und der Nächste Nachbar. *Gesucht:* je Netz und Zelle (Größe, Familie, Wechsel) der Unterschied der Tourlänge zum
Nächsten Nachbarn und das Urteil. Der Block ist ein vollständiges Modell der **Messung** (Instanzen, Wechsel, Netze im Einsatz, Urteil, Kosten); das Training steht nur als Politikgradient mit Referenzwert
(vollständig in der Demo zum Zeigernetz und zur Zugwahl, Kopien `nfi_net.py`, `nfi_train.py`, `nfik_policy.py`). Instanzmodell: drei Familien aus `nfi_scenario.py`, Zufalls-Seed 0 bis 99; Zufallsquelle SplitMix64.

| Symbol | Bedeutung |
|---|---|
| n | Zahl der Orte; x_i ∈ ℝ² Koordinaten von Ort i; X ∈ ℝ^(n×2) |
| π | Tour (Permutation), π(1) = 0 (Start Stadt 0) |
| L(π) | Tourlänge, d_ij = ‖x_i − x_j‖ |
| L_ref | Referenz (n ≤ 20 exakt, sonst beste bekannte Lösung) |
| a > 0, b ∈ ℝ² | Faktor und Verschiebung des Wechsels |
| m, r | Minimum je Achse und größte Spannweite (Normierung) |
| d, C | Einbettungsbreite 32 des Zeigernetzes, Beschneidungskonstante 10 |
| H, c_t, u_j | Kodiererausgabe, Kontext im Schritt t, Zeiger-Wert des freien Ortes j |
| s, s0 | Skala der Instanz (mittlerer Abstand zum nächsten Nachbarn), feste Skala 0,1239 |
| x_ij ∈ ℝ⁷ | Merkmale des Kandidaten j von Ort i (Merkmalsnetz) |
| Δ | Unterschied Netz minus Nächster Nachbar in Prozentpunkten der Referenz |
| K, O | Kandidatenbewertungen, grobe Operationen |"""

FORMELN = [
    ("Tourlänge und Unterschied:", r"L(\pi)=\sum_{t=1}^{n}\lVert x_{\pi(t)}-x_{\pi(t+1)}\rVert,\qquad \pi(n+1)=\pi(1)"),
    ("", r"g=100\,\frac{L-L_{\mathrm{ref}}}{L_{\mathrm{ref}}},\qquad \Delta=100\,\frac{L_{\mathrm{Netz}}-L_{\mathrm{NN}}}{L_{\mathrm{ref}}}"),
    ("Skalen- und Lagewechsel einer Instanz (die Länge skaliert, der Nächste Nachbar bleibt):", r"x_i\;\mapsto\;a\,x_i+b\quad(a>0),\qquad L\;\mapsto\;a\,L"),
    ("Normierung der Eingabe (invariant gegen Verschieben und positives Skalieren, erhält das Seitenverhältnis):",
     r"x_i'=\frac{x_i-m}{r},\quad m=\bigl(\min_i x_{i1},\min_i x_{i2}\bigr),\quad r=\max\bigl(\max_i x_{i1}-m_1,\max_i x_{i2}-m_2\bigr)"),
    ("", r"\text{normiere}(a\,x+b)=\text{normiere}(x)\quad(a>0)"),
    ("Zeigernetz, Kodierer (eine Schicht, mit Rückgriff auf die Eingabe):", r"H^{0}=X\,W_0+b_0,\qquad A=\operatorname{softmax}\!\Bigl(\tfrac{(H^0W_q)(H^0W_k)^{\top}}{\sqrt d}\Bigr),\qquad H^{1}=H^{0}+A\,(H^0W_v)\,W_o"),
    ("", r"H=H^{1}+\operatorname{ReLU}(H^{1}W_1+b_1)\,W_2+b_2"),
    ("Dekodierer im Schritt t (aktueller Ort π(t), freie Orte F_t):",
     r"c_t=\bigl[\bar h;\,h_{\pi(1)};\,h_{\pi(t)}\bigr]W_c,\qquad u_j=C\,\tanh\!\Bigl(\tfrac{c_t\cdot(h_jW_{kd})}{\sqrt d}\Bigr),\qquad p_t(j)=\frac{e^{u_j}}{\sum_{k\in F_t}e^{u_k}}\ \ (j\in F_t)"),
    ("Sättigung des Zeigers (Mechanismus von Faktor 10): für große |z| ist tanh z nahe ±1, die Unterschiede verschwinden:", r"1-\tanh z=\frac{2}{e^{2z}+1}\;\approx\;2e^{-2z}"),
    ("Merkmalsnetz (kein Kodierer): Politik über die freien Orte, sieben Merkmale je Kandidat j von Ort i; die Merkmale 1, 3, 4 und 7 sind bei 6 gekappt (die Kappung steht der Kürze halber nicht in der Formel):",
     r"z_{ij}=w_2^{\top}\tanh(W_1x_{ij}+b_1)+b_2,\qquad p(j)=\frac{e^{z_{ij}}}{\sum_{k\in F}e^{z_{ik}}}\quad(W_1\in\mathbb R^{16\times7})"),
    ("", r"x_{ij}=\bigl(\tfrac{d_{ij}}{s},\ \tfrac{\min(\rho_{ij},8)}{8},\ \tfrac{d_{j,\pi(1)}}{s},\ \tfrac{\min_{k\in F\setminus j}d_{jk}}{s},\ \tfrac{\min(|\{k\in F\setminus j:\,d_{jk}\le 2s\}|,8)}{8},\ \tfrac{|V|}{n},\ \tfrac{d_{ij}}{\min_{k\in F}d_{ik}}-1\bigr)"),
    ("Dabei ist ρ_ij der Rang von j unter den freien Nachbarn von i nach Entfernung (0 für den nächsten) und V die Menge der besuchten Orte. Skala der Instanz und feste Skala (Schalter „mit s / ohne s“):",
     r"s=\frac1n\sum_{i}\min_{j\ne i}d_{ij},\qquad s\;\leftarrow\;s_0=0{,}1239\ \text{ (ohne s)}"),
    ("Training (nur in `tools/`, nicht in der App): Politikgradient mit Referenzwert aus dem Mittel,", r"\nabla_\theta J=\mathbb{E}\bigl[(b-L)\,\nabla_\theta\log p_\theta(\pi)\bigr],\qquad b=\text{Mittel der Chargen- bzw. Rollout-Längen}"),
    ("Urteilsregel je Zelle (Instanzen i = 1 … M, Zufalls-Seeds q = 1 … 5, Intervall 95 % Bootstrap über Instanzen):",
     r"\text{Urteil}=\begin{cases}\text{bricht}&\bar\Delta>0,\ \text{Intervall}\subset(0,\infty),\ \#\{q:\bar\Delta_q>0\}\ge4\\ \text{schlägt}&\bar\Delta<0,\ \text{Intervall}\subset(-\infty,0)\\ \text{hält}&\text{sonst}\end{cases}"),
    ("Kosten je Konstruktion (Annahmen: Multiplikation-Addition = 1, Nächster Nachbar 8 Operationen je Kandidat):", r"K=\tfrac{n(n-1)}{2},\qquad O_{\mathrm{Zeiger}}=2nd+8nd^2+2n^2d+nd^2+2d^2+(n-1)(d^2+nd+4n)"),
    ("", r"O_{\mathrm{Merkmal}}=(n-1)\bigl(2n^2+n(8+\log_2 n)\bigr)+\tfrac{n(n-1)}{2}\cdot144,\qquad O_{\mathrm{NN}}=8\,\tfrac{n(n-1)}{2}"),
]

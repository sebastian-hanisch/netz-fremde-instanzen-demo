"""Konstanten: Netze, Training, Instanzen, Wechsel, Zufalls-Seeds je Zweck, Marken der Urteile, Zählannahmen, Regler-Grenzen, Farben und Presets.

Die Namen und Werte im ersten Block (C_CLIP, BATCH, LERNRATE, Zwecke, Instanzkonstanten, Marken, Modellformat) sind Schnittstelle zu den kopierten Dateien aus Stück 1
(nfi_net, nfi_train, nfi_scenario, nfi_instances, nfi_stats, nfi_cost); sie stehen hier mit denselben Namen und Werten wie in `znz_constants.py` (Test `tests/test_copies.py`).
Der zweite Block gehört dieser Demo."""

# ================================================================ Schnittstelle der Kopien (Werte wie znz_constants)
C_CLIP = 10.0                              # Beschneidung der Zeigerwerte: u = C tanh(s / sqrt(d))
BATCH = 120                                # Touren je Trainingsschritt des Zeigernetzes
LERNRATE = 1e-3                            # Adam des Zeigernetzes (nur für d = 32 und n = 20 abgestimmt)
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
NORM_BEGRENZUNG = 1.0                      # Euklid-Norm des Gradienten wird auf 1 begrenzt
EVAL_ALLE = 100
LETZTE_MESSPUNKTE = 3
FAMILIEN = ("gleich", "cluster", "gitter")  # Index = Schlüssel der Zufalls-Seeds
FAMILIE_HAND = "hand"
SEITE = 1000.0                              # Quadrat 1000 x 1000; das Netz sieht Koordinaten durch 1000, also das Einheitsquadrat
CLUSTER_ZAHL, CLUSTER_SIGMA = 5, 40.0
GITTER_FAKTOR, GITTER_RAUSCHEN = 1.5, 0.1
VAL_OFFSET = 1000                           # Validierungsinstanzen haben Zufalls-Seeds ab 1000
HAND_PUNKTE = ((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0), (2.0, 1.5))   # (nur von der kopierten hand_instance gebraucht, hier ungenutzt)
HAND_SKALA = 4.0
ZWECK_INIT, ZWECK_TRAIN_X, ZWECK_TRAIN_S, ZWECK_SAMPLE, ZWECK_BOOT, ZWECK_ZUFALL = 2, 3, 4, 5, 6, 8
ZWECK_INSTANZ, ZWECK_START, ZWECK_LAUF = 11, 33, 24     # Werte der Quelle hhw_scenario
MARKE_KLAR, MARKE_GLEICH = 0.25, 0.10
BOOTSTRAP_ZIEHUNGEN = 2000
OPS_JE_KANDIDAT = 8                        # Annahme: grobe Operationen je Kandidatenbewertung des Nächsten Nachbarn
MODELL_FORMAT = "znz-modell-1"

# ================================================================ Eigene Zwecke der Zufalls-Seeds (Bereich nfi 50 bis 99; znz 0 bis 49, zwl 100 bis 149)
ZWECK_AUG = 53                             # Augmentierung im Training (Faktor und Verschiebung je Instanz)
ZWECK_MISCH = 54                           # Wahl der Größe je Schritt beim Mischtraining
ZWECK_BOOT_ZELLE = 71                      # Bootstrap-Intervall einer Zelle
ZWECK_TEILPROBE = 72                       # Teilstichproben in E4
ZWECK_WAHL = 73                            # Reserve (nicht belegt)

# ================================================================ Netze und Trainingsrezepte (alle bei n = 20, gleichverteilt)
D_ZEIGER = 32
ZEIGER_PARAMS = 12480
MERKMAL_PARAMS = 145
SCHRITTE_ZEIGER, SCHRITTE_MERKMAL = 8000, 3750
TOUREN_ZEIGER = SCHRITTE_ZEIGER * BATCH            # 960 000
TOUREN_MERKMAL = 3750 * 256                        # 960 000
ZEILEN_MERKMAL = TOUREN_MERKMAL * 19 * 20 // 2     # 182 400 000 Merkmalszeilen (n(n-1)/2 je Tour, n = 20)
N_TRAIN = 20
NETZ_SEEDS = 5                                     # Gewichts-Zufalls-Seeds 0 bis 4 (gepaart: Gewichte und Trainingsinstanzen)
S0 = 0.1239                                       # feste Skala „ohne s“: Mittel von s über 20 000 Trainingsinstanzen bei n = 20 (Strom ZWECK_TRAIN_X, Daten-Seed 99): 0,123884; die Vormessung hatte 0,1236
AUG_FAKTOR = (0.1, 10.0)                           # Aaug: Faktor logarithmisch gleichverteilt
AUG_VERSCHIEBUNG = (-3.0, 3.0)                     # Aaug: Verschiebung gleichverteilt je Achse
MISCH_GROESSEN = (10, 20, 50)                      # Amix: Größe je Schritt gleichverteilt
MODELL_ORDNER = "nfi_models"
NETZE = ("A", "Ant", "An", "Bmit", "Bohne", "Aaug", "Amix")
NETZE_BASIS = ("A", "Ant", "An", "Bmit", "Bohne")  # E1 bis E4
NETZ_ANZEIGE = {
    "A": "Zeigernetz, rohe Eingabe",
    "Ant": "Zeigernetz, Eingabe beim Einsatz normiert",
    "An": "Zeigernetz, Eingabe auch im Training normiert",
    "Bmit": "Merkmalsnetz mit Skala s der Instanz",
    "Bohne": "Merkmalsnetz mit fester Skala s0",
    "Aaug": "Zeigernetz, im Training skaliert und verschoben",
    "Amix": "Zeigernetz, normiert, im Training n = 10, 20, 50",
}
NETZ_KURZ = {"A": "A", "Ant": "A+nt", "An": "An", "Bmit": "B mit s", "Bohne": "B ohne s", "Aaug": "Aaug", "Amix": "Amix"}
ZEIGER_NETZE = ("A", "Ant", "An", "Aaug", "Amix")
MERKMAL_NETZE = ("Bmit", "Bohne")
# Netze, die unter Skala und Lage invariant sind (Konstruktion: normierte Eingabe bzw. Skala s der Instanz): sie werden unter den Wechseln nicht noch einmal gerechnet
INVARIANTE_NETZE = ("Ant", "An", "Bmit", "Amix")
WECHSEL_NETZE = ("A", "Bohne", "Aaug")             # in der eingefrorenen Messung unter den Wechseln gerechnet (Aaug nur mit Mischtraining)

# ================================================================ Instanzen und Wechsel
FAMILIE_ANZEIGE = {"gleich": "gleichverteilt", "cluster": "geclustert", "gitter": "Gitter"}
GROESSEN = (10, 20, 50, 100, 200)
WECHSEL_GROESSEN = (20, 50)                        # Skalen- und Lagewechsel nur bei diesen n
WECHSEL = ("basis", "klein", "gross", "versch")
WECHSEL_ANZEIGE = {"basis": "Basis", "klein": "Faktor 0,25", "gross": "Faktor 10", "versch": "Verschiebung (3; 2)"}
FAKTOR_KLEIN, FAKTOR_GROSS = 0.25, 10.0
VERSCHIEBUNG = (3.0, 2.0)
# Instanzen je Größe in der eingefrorenen Messung (Zufalls-Seed 0 bis Anzahl − 1, Familie je Größe gleich viele); im Bau nach Rechenlast gesetzt, siehe README
INSTANZEN = {10: 100, 20: 100, 50: 100, 100: 100, 200: 50}
GLS_SEKUNDEN = {50: 2, 100: 4, 200: 6}             # Zeitlimit der Referenzlösung (Guided Local Search, OR-Tools, nur Werkzeug) je Instanz und Größe
DIAGNOSE_INSTANZEN = 30                            # Instanzen der Diagnosen (n = 20 gleichverteilt)
NN2OPT_INSTANZEN_GROSS = 10                        # NN + 2-opt bei n = 200 nur für die ersten Instanzen (reines Python)
TEIL_N = 20                                        # Größe der Teilmenge (live) in E1: höchstens 30 Instanzen

# ================================================================ Urteile (gesetzt, nicht abgeleitet)
SEEDS_MIN_BRUCH = 4                                # „bricht“ braucht mindestens 4 von 5 Zufalls-Seeds über null
MARKE_HAELT_LUECKE = 2.0                           # Regel (b) der Vormessung: Lückenanstieg unter 2 Prozentpunkten gilt als „hält die Basislücke“
MISCH_AM_TRAINING_BASIS, MISCH_AM_TRAINING_WECHSEL, MISCH_AN_EINGABE = 1, 3, 4   # Schwellen von E5 (vor der Messung gesetzt)

# ================================================================ Regler
SEED_MAX = 99                                      # Zufalls-Seed der Instanz: 0 bis 99
N_STUFEN = (5, 10, 20, 50, 100, 200)               # 5: nur das Handbeispiel
N_HAND = 5
FAKTOR_STUFEN = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0)
VERSCHIEBUNG_MIN, VERSCHIEBUNG_MAX = -5, 5
NORM_MODI = ("aus", "einsatz", "training")          # A / A+nt / An
NORM_ANZEIGE = {"aus": "aus (Zeigernetz sieht rohe Koordinaten)", "einsatz": "beim Einsatz", "training": "auch im Training"}
SKALA_MODI = ("mit", "ohne")
SKALA_ANZEIGE = {"mit": "Skala s der Instanz", "ohne": "feste Skala s0"}
SEEDS_ZAHL_MAX = 5
LIVE_O_MAX = 1.0e9                                 # Obergrenze grobe Operationen je Klick
LIVE_M_MAX = 30                                    # Instanzen live in E1
E2_KNOPF_M_MAX = 6                                 # Instanzen des Einzelknopfs in E2
E4_ZIEHUNGEN_MAX = 4_000_000                       # R · 400 · m
E4_BOOT = 400
E4_R_MIN, E4_R_MAX, E4_R_STANDARD = 50, 300, 200
DEFAULT_FAMILIE, DEFAULT_N, DEFAULT_SEED = "gleich", 20, 0
DEFAULT_FAKTOR, DEFAULT_VX, DEFAULT_VY = 10.0, 0, 0
DEFAULT_NORM, DEFAULT_SKALA, DEFAULT_NETZ_SEED, DEFAULT_SEEDS_ZAHL = "aus", "mit", 0, 5
DEFAULT_E1_M = 12
DEFAULT_E4_M = 10

# ================================================================ Farben
COLOR_NN = "#e45756"
COLOR_A = "#2F6B65"
COLOR_B = "#4c78a8"
COLOR_OPT = "#54a24b"
COLOR_2OPT = "#f58518"
COLOR_GRAU = "rgba(150,150,150,0.75)"
COLOR_BAND = "rgba(47,107,101,0.18)"
NETZ_FARBE = {"A": "#2F6B65", "Ant": "#54a24b", "An": "#9d755d", "Bmit": "#4c78a8", "Bohne": "#b279a2", "Aaug": "#f58518", "Amix": "#e0bf3a"}

# ================================================================ Presets (das erste ist die Voreinstellung beim Öffnen)
_BASIS = {
    "familie": DEFAULT_FAMILIE, "n": DEFAULT_N, "seed": DEFAULT_SEED, "faktor": DEFAULT_FAKTOR, "vx": DEFAULT_VX, "vy": DEFAULT_VY,
    "norm": DEFAULT_NORM, "skala": DEFAULT_SKALA, "netz_seed": DEFAULT_NETZ_SEED, "seeds_zahl": DEFAULT_SEEDS_ZAHL,
}
PRESETS = {
    "Faktor 10, Zeigernetz roh": dict(_BASIS),
    "Handbeispiel: fünf Orte": {**_BASIS, "n": 5, "faktor": 1.0},
    "Faktor 10, Eingabe normiert": {**_BASIS, "norm": "einsatz"},
    "Merkmalsnetz ohne s, Faktor 10": {**_BASIS, "norm": "einsatz", "skala": "ohne"},
    "Größe: n = 100": {**_BASIS, "n": 100, "faktor": 1.0, "norm": "einsatz"},
    "Verteilung: Gruppen, n = 50": {**_BASIS, "familie": "cluster", "n": 50, "faktor": 1.0},
    "Sehr groß: n = 200": {**_BASIS, "n": 200, "faktor": 1.0, "norm": "einsatz", "skala": "ohne"},
}
PRESET_HELP = {
    "Faktor 10, Zeigernetz roh": "20 Orte gleichverteilt, dieselbe Instanz zehnmal größer gemessen: das Zeigernetz bekommt rohe Koordinaten (Eingabe nicht normiert), das Merkmalsnetz benutzt die Skala s der Instanz.",
    "Handbeispiel: fünf Orte": "Fünf Orte zum Nachrechnen (Nächster Nachbar und exakte Tour, Entfernungen und Merkmale von Hand in der Lupe von Experiment 1). Keine Netzläufe: die Netze sind bei n = 20 trainiert.",
    "Faktor 10, Eingabe normiert": "Wie das erste Szenario, aber die Eingabe des Zeigernetzes wird beim Einsatz auf das Einheitsquadrat normiert: dieselbe Tour wie auf der Basisinstanz (folgt aus der Konstruktion).",
    "Merkmalsnetz ohne s, Faktor 10": "Faktor 10, das Merkmalsnetz benutzt die feste Skala s0 statt der Skala der Instanz: seine Entfernungsmerkmale laufen in die Kappung. Live mit 12 Instanzen; die eingefrorene Messung (100 Instanzen) kann im Urteil abweichen.",
    "Größe: n = 100": "100 Orte gleichverteilt, das Netz ist bei n = 20 trainiert; Zeigernetz mit normierter Eingabe und Merkmalsnetz mit s (eingefrorene Messung in Experiment 2).",
    "Verteilung: Gruppen, n = 50": "50 Orte in Gruppen (fünf Haufen), Zeigernetz roh und Merkmalsnetz mit s; die Normierung ändert hier das Urteil des Zeigernetzes (Experiment 3).",
    "Sehr groß: n = 200": "200 Orte gleichverteilt: Zeigernetz mit normierter Eingabe und Merkmalsnetz mit fester Skala; breite Intervalle, wenige Instanzen (Experiment 2).",
}

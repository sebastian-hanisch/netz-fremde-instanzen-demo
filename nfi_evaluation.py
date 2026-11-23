"""Auswertung: Zahlenformate, die Urteilsregel `klasse_zelle` und alle Verdikte als reine Funktionen (die App ruft `getattr(st, art)(text)`).

Jedes Verdikt gibt `(art, text)` zurück; art: "success", "info" oder "warning". Alle Schwellen sind gesetzt, nicht abgeleitet (`nfi_constants`).

Urteilsregel je Zelle (Instanzen i = 1 bis M, Zufalls-Seeds q = 1 bis S, Unterschied Delta = 100 (L_Netz - L_NN) / L_ref in Prozentpunkten, gepaart über Instanzen,
Intervall 95 % Bootstrap über Instanzen):
* **bricht**: Unterschied > 0, Intervall schließt 0 aus und mindestens 80 % der Zufalls-Seeds (aufgerundet; bei S = 5: 4 von 5) liegen mit ihrem mittleren Unterschied über 0;
* **schlägt**: Unterschied < 0 und Intervall schließt 0 aus;
* **hält**: sonst; liegt 0 im Intervall, heißt das im Text „nicht unterscheidbar“.
Jeder Text nennt Instanzzahl, Zahl der Zufalls-Seeds und die Voraussetzung (Training nur bei n = 20 gleichverteilt, eine greedy-Tour je Instanz, Latte Nächster Nachbar)."""

import math

import nfi_constants as C

KLASSE_ANZEIGE = {"bricht": "bricht unter dem Nächsten Nachbarn", "haelt": "hält den Nächsten Nachbarn", "schlaegt": "schlägt den Nächsten Nachbarn"}
KLASSE_KURZ = {"bricht": "bricht", "haelt": "hält", "schlaegt": "schlägt"}
NETZ_NAME = {"zeiger": "Zeigernetz", "merkmal": "Merkmalsnetz"}
MIN_INSTANZEN_URTEIL = 10


# ---------------------------------------------------------------------------------------------------------------- Zahlenformate (deutsch)
def de(x: float, stellen: int = 2) -> str:
    """Zahl mit Dezimalkomma (eine Zahl, die auf null rundet, ohne Minuszeichen)."""
    s = f"{x:.{stellen}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace(".", ",")


def de_vz(x: float, stellen: int = 2) -> str:
    """Zahl mit Dezimalkomma und Vorzeichen (eine Zahl, die auf null rundet, ohne Vorzeichen)."""
    s = f"{x:+.{stellen}f}"
    if float(s) == 0:
        s = s[1:]
    return s.replace(".", ",")


def zahl(n) -> str:
    """Ganzzahl mit Leerzeichen als Tausendertrenner (1 850)."""
    return f"{int(round(n)):,}".replace(",", " ")


def faktor(x: float) -> str:
    """Faktor „das 162-Fache“: ab 10 ohne Nachkommastelle, darunter mit einer."""
    return de(x, 0) if x >= 10 else de(x, 1)


def wiss(x: float) -> str:
    """Kleine Zahlen in Exponentenschreibweise mit Dezimalkomma (1,0e-11), sonst drei Nachkommastellen."""
    if abs(x) >= 1e-3 or x == 0:
        return de(x, 3)
    return f"{x:.1e}".replace(".", ",")


def intervall(lo, hi, stellen: int = 1) -> str:
    """„[−2,1; +0,4]“ (Intervall in Prozentpunkten)."""
    return f"[{de_vz(lo, stellen)}; {de_vz(hi, stellen)}]"


# ---------------------------------------------------------------------------------------------------------------- Urteilsregel
def seeds_noetig(seeds: int) -> int:
    """Zahl der Zufalls-Seeds über null, die „bricht“ braucht: 80 % aufgerundet (bei 5 Seeds: 4)."""
    return max(1, math.ceil(0.8 * seeds - 1e-9))


def klasse_zelle(diff: float, lo: float, hi: float, seeds_pos: int, seeds: int = C.NETZ_SEEDS) -> str:
    """Urteil einer Zelle: "bricht", "schlaegt" oder "haelt" (Regeln im Kopf der Datei)."""
    if diff > 0 and lo > 0 and seeds_pos >= seeds_noetig(seeds):
        return "bricht"
    if diff < 0 and hi < 0:
        return "schlaegt"
    return "haelt"


def nicht_unterscheidbar(lo, hi) -> bool:
    """Liegt 0 im Intervall?"""
    return lo is not None and hi is not None and lo <= 0 <= hi


def zaehle_klassen(klassen) -> dict:
    """{"bricht": k, "haelt": h, "schlaegt": s, "gesamt": z} über eine Liste von Urteilen (None zählt nicht)."""
    ks = [k for k in klassen if k is not None]
    return {"bricht": ks.count("bricht"), "haelt": ks.count("haelt"), "schlaegt": ks.count("schlaegt"), "gesamt": len(ks)}


def urteil_wort(z: dict) -> str:
    """Urteil einer Zelle mit Zusatz „nicht unterscheidbar“ bei Intervall über 0 (z aus `nfi_eval.zelle`)."""
    if z["klasse"] is None:
        return "kein Urteil"
    w = KLASSE_KURZ[z["klasse"]]
    if z["klasse"] == "haelt" and nicht_unterscheidbar(z["lo"], z["hi"]):
        w += ", nicht unterscheidbar"
    return w


def zellen_text(z: dict) -> str:
    """„+4,4 [+2,0; +6,5], 5 von 5 Zufalls-Seeds darüber, 100 Instanzen“."""
    return f"{de_vz(z['v'], 1)} {intervall(z['lo'], z['hi'])}, {z['seeds_pos']} von {z['seeds']} Zufalls-Seeds darüber, {z['m']} Instanzen"


# ---------------------------------------------------------------------------------------------------------------- Kosten
def kosten_zeile(n: int, k: int, f_zeiger: float, f_merkmal: float) -> str:
    """Zusatz zu jedem Verdikt, das ein Netz mit dem Nächsten Nachbarn vergleicht (Zählregel der Linie)."""
    return (f"Eine Konstruktion je Instanz: K = {zahl(k)} für alle drei gleich; in O kostet das Zeigernetz das {faktor(f_zeiger)}-Fache, "
            f"das Merkmalsnetz das {faktor(f_merkmal)}-Fache des Nächsten Nachbarn (n = {n}).")


# ---------------------------------------------------------------------------------------------------------------- Verdikte
def verdict_instanz(netz_name: str, d: float, einheit: str = "Prozentpunkte") -> tuple:
    """E0: eine Instanz. `d` = Länge des Netzes minus Länge des Nächsten Nachbarn in Prozentpunkten der Referenz (negativ: Netz kürzer); ohne Referenz in Prozent der Länge des Nächsten Nachbarn
    (`einheit` sagt es)."""
    richtung = "kürzer" if d < 0 else "länger"
    return "info", (f"Auf dieser einen Instanz ist das {netz_name} um {de(abs(d), 1)} {einheit} {richtung} als der Nächste Nachbar. "
                    "Eine Instanz sagt über die Zellen unten nichts; die Aussagen stehen in den Abschnitten darunter.")


def verdict_skala(faktor_: float, vx: float, vy: float, ergebnisse: list, m: int, kosten: str = "") -> tuple:
    """E1: ein Wechsel. `ergebnisse`: Liste von {"name", "z" (Zelle aus `nfi_eval.zelle`), "invariant" (bool)}.

    Art: info bei weniger als 10 Instanzen, warning wenn ein gewähltes Netz bricht, sonst success."""
    basis = abs(faktor_ - 1.0) < 1e-12 and vx == 0 and vy == 0
    kopf = "Keine Veränderung: das ist die Basisinstanz." if basis else f"Bei Faktor {de(faktor_, 2)} und Verschiebung ({de(vx, 0)}; {de(vy, 0)}):"
    saetze = []
    for e in ergebnisse:
        z = e["z"]
        if z["klasse"] is None:
            saetze.append(f"für das {e['name']} gibt es mit {z['m']} Instanz kein Intervall.")
            continue
        zusatz = " (nicht unterscheidbar)" if z["klasse"] == "haelt" and nicht_unterscheidbar(z["lo"], z["hi"]) else ""
        satz = f"das {e['name']} {KLASSE_ANZEIGE[z['klasse']]}{zusatz}: Unterschied {zellen_text(z)}."
        if e.get("invariant"):
            satz += " Dieses Netz sieht die Instanz nach dem Normieren unverändert; die Gleichheit mit der Basiszelle folgt aus der Konstruktion und sagt nichts über Lernen."
        saetze.append(satz)
    text = kopf + " " + " ".join(saetze) + (" " + kosten if kosten else "")
    if m < MIN_INSTANZEN_URTEIL:
        return "info", f"Mit {m} Instanzen ist die Stichprobe klein (mindestens {MIN_INSTANZEN_URTEIL} für ein Urteil dieser Demo). " + text
    art = "warning" if any(e["z"]["klasse"] == "bricht" for e in ergebnisse) else "success"
    return art, text


def verdict_groesse(familie: str, a_name: str, b_name: str, a_klassen: dict, b_klassen: dict, instanzen: dict, n_zeile: int, g_nn2opt, g_b, kosten: str = "") -> tuple:
    """E2: Größen. `a_klassen`, `b_klassen`: {n: Urteil} je Größe (n = 10 bis 200), `instanzen`: {n: Zahl der Instanzen}; `g_nn2opt`, `g_b`: Lücken in Prozent bei n = `n_zeile` (None: nicht gemessen)."""
    gs = sorted(a_klassen)
    z = len(gs)
    ka = [n for n in gs if a_klassen[n] == "bricht"]
    kb = [n for n in gs if b_klassen[n] == "bricht"]
    sb = [n for n in gs if b_klassen[n] == "schlaegt"]
    teil_a = (f"das {a_name} bricht unter dem Nächsten Nachbarn in {len(ka)} von {z} Größen" + (f" (zuerst bei n = {ka[0]})" if ka else ""))
    teil_b = f"das {b_name} bricht in {len(kb)} von {z} Größen, schlägt ihn in {len(sb)}"
    inst = ", ".join(str(instanzen[n]) for n in gs)
    text = (f"Familie {familie}, Training bei n = {C.N_TRAIN}: {teil_a}; {teil_b}. Instanzen je Größe (n = {', '.join(str(n) for n in gs)}): {inst}; bei großem n sind die Intervalle breit.")
    if g_nn2opt is not None and g_b is not None:
        text += f" Nächster Nachbar ist eine niedrige Latte: Nächster Nachbar mit 2-opt liegt bei n = {n_zeile} bei {de(g_nn2opt, 1)} % Lücke gegen {de(g_b, 1)} % beim {b_name}."
    if kosten:
        text += " " + kosten
    return ("warning" if (ka or kb) else "success"), text


def verdict_verteilung(n: int, a_name: str, b_name: str, a_klassen: dict, b_klassen: dict, a_roh: dict, a_norm: dict, kosten: str = "", a_norm_klassen: dict | None = None) -> tuple:
    """E3: Verteilung. `a_klassen`, `b_klassen`: {Familie: Urteil}; `a_roh`, `a_norm`: {Familie: Unterschied in Prozentpunkten} des Zeigernetzes ohne und mit Normierung beim Einsatz,
    `a_norm_klassen`: dessen Urteile mit Normierung. Die Normierung ändert das Urteil in den Familien, in denen beide Urteile verschieden sind."""
    fams = list(a_klassen)
    ka = [f for f in fams if a_klassen[f] == "bricht"]
    kb = [f for f in fams if b_klassen[f] == "bricht"]
    text = f"Bei n = {n}: das {a_name} bricht in {len(ka)} von {len(fams)} Familien" + (f" ({', '.join(ka)})" if ka else "") + f", das {b_name} in {len(kb)}" + (f" ({', '.join(kb)})" if kb else "") + "."
    aend = [f for f in fams if a_norm_klassen is not None and a_norm_klassen.get(f) != a_klassen[f]]
    if aend:
        f = aend[0]
        text += f" Die Normierung ändert bei {f} das Urteil des Zeigernetzes: ja ({de_vz(a_roh[f], 2)} gegen {de_vz(a_norm[f], 2)} Prozentpunkte)."
    else:
        text += " Die Normierung ändert in keiner der Familien das Urteil des Zeigernetzes."
    if kosten:
        text += " " + kosten
    return ("warning" if (ka or kb) else "success"), text


def verdict_stichprobe(m: int, a: int, R: int, M: int, urteil: str, h: float, n: int) -> tuple:
    """E4: Teilstichproben. In `a` von `R` Teilstichproben vom Umfang `m` kommt dasselbe Urteil heraus wie mit allen `M` Instanzen; `h` = mittlere halbe Intervallbreite."""
    if m >= M:
        return "info", f"Teilstichprobe gleich Vollstichprobe: alle {M} Instanzen der Zelle, Urteil {urteil}."
    return "info", (f"Mit {m} Instanzen kommt in {a} von {R} Teilstichproben dasselbe Urteil heraus wie mit allen {M} Instanzen der Zelle ({urteil}); "
                    f"das Intervall ist im Mittel ±{de(h, 1)} Prozentpunkte breit. Bei n = {n} sind {M} Instanzen gemessen.")


def verdict_mischtraining(k_basis: int, k_basis_ohne: int, kk_gross: int, z_gross: int, ka_wechsel: int, z_wechsel: int, k200: int, z200: int, ursache: str,
                          aaug_seeds: int | None = None, seeds: int = C.NETZ_SEEDS) -> tuple:
    """E5: Mischtraining. Die Aussage zur Ursache steht nur, wenn die Zählung sie belegt (Schwellen in `nfi_constants`, vor der Messung gesetzt); `ursache` aus `ursache_satz`.
    `aaug_seeds`: Zahl der Zufalls-Seeds des augmentiert trainierten Netzes unter dem Nächsten Nachbarn auf der Trainingszelle (Modellgüte-Marke: unter 4 von 5 heißt „schwach“)."""
    text = (f"Basiszellen: das Zeigernetz mit gemischtem Training (Größen 10, 20, 50, normierte Eingabe) bricht in {k_basis} von 15 Zellen, das Zeigernetz mit normierter Eingabe ohne Mischtraining in {k_basis_ohne}; "
            f"ab n = 100 {kk_gross} von {z_gross}. Skala und Lage: mit Augmentierung bricht es in {ka_wechsel} von {z_wechsel} Zellen, mit Normierung in 0. "
            f"Bei n = 200, das nicht im Training vorkam, bricht es in {k200} von {z200} Zellen.")
    if aaug_seeds is not None and aaug_seeds < C.SEEDS_MIN_BRUCH:
        text += (f" Das Netz mit Augmentierung ist schwach: nur {aaug_seeds} von {seeds} Zufalls-Seeds liegen auf der Trainingszelle (n = 20 gleichverteilt) unter dem Nächsten Nachbarn; "
                 "seine Zählung sagt nichts über Augmentierung im Allgemeinen, nur über dieses Training.")
    return "info", text + " " + ursache


def ursache_satz(amix_gross_bruch: int, aaug_wechsel_bruch: int, aaug_schwach: bool = False) -> str:
    """Satz zur Ursache der Brüche des Zeigernetzes; nur mit Zählung, sonst „nicht entschieden“ (Schwellen gesetzt, nicht abgeleitet). Ein schwaches augmentiertes Netz belegt „am Training“ nie."""
    if amix_gross_bruch <= C.MISCH_AM_TRAINING_BASIS and aaug_wechsel_bruch <= C.MISCH_AM_TRAINING_WECHSEL and not aaug_schwach:
        return "Gemessen für diese Netze: die Brüche hängen am Training bei einer einzigen Größe und Skala, nicht an der Eingabedarstellung."
    if amix_gross_bruch >= C.MISCH_AN_EINGABE:
        return ("Gemessen für diese Netze: mit Training bei n = 10, 20 und 50 und normierter Eingabe bricht das Zeigernetz ab n = 100 weiterhin in mindestens 4 von 6 Zellen; "
                "das spricht in dieser Rezeptur gegen „nur das Training bei einer einzigen Größe“. Ob andere Rezepte oder größere Trainingsgrößen es ändern, ist nicht gemessen.")
    return "Ob die Brüche an der Eingabedarstellung oder am Training bei einer Größe hängen, ist mit dieser Messung nicht entschieden."

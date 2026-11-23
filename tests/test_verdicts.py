"""Verdikte als reine Funktionen (`nfi_evaluation.py`): Schwellen und jede Zahl im Text an handgesetzten Zeilen; Nullwächter und Randfälle."""

import pytest

import nfi_constants as C
import nfi_evaluation as V


def _z(v, lo, hi, k=5, S=5, m=100, klasse="auto", luecke=10.0):
    return {"v": v, "lo": lo, "hi": hi, "seeds_pos": k, "seeds": S, "m": m, "luecke": luecke,
            "klasse": V.klasse_zelle(v, lo, hi, k, S) if klasse == "auto" else klasse}


# ---------------------------------------------------------------- klasse_zelle
@pytest.mark.parametrize("v,lo,hi,k,soll", [
    (4.45, 2.07, 6.62, 5, "bricht"), (4.45, 2.07, 6.62, 4, "bricht"), (4.45, 2.07, 6.62, 3, "haelt"),      # 4 von 5 genügt, 3 von 5 nicht
    (4.45, -0.01, 6.62, 5, "haelt"),                                                                       # Intervall mit 0
    (-5.51, -8.18, -2.98, 0, "schlaegt"), (-1.11, -2.88, 0.63, 0, "haelt"), (0.0, -1.0, 1.0, 2, "haelt"),
])
def test_klasse_zelle_schwellen(v, lo, hi, k, soll):
    assert V.klasse_zelle(v, lo, hi, k, 5) == soll


def test_klasse_zelle_bei_weniger_seeds_braucht_alle_aufgerundet():
    assert V.klasse_zelle(1.0, 0.5, 1.5, 1, 1) == "bricht" and V.klasse_zelle(1.0, 0.5, 1.5, 1, 2) == "haelt" and V.klasse_zelle(1.0, 0.5, 1.5, 2, 2) == "bricht"


def test_nicht_unterscheidbar_nur_bei_intervall_ueber_null():
    assert V.nicht_unterscheidbar(-1.0, 1.0) and not V.nicht_unterscheidbar(0.1, 1.0) and not V.nicht_unterscheidbar(None, None)


def test_zaehle_klassen_ignoriert_none():
    assert V.zaehle_klassen(["bricht", "haelt", None, "schlaegt", "bricht"]) == {"bricht": 2, "haelt": 1, "schlaegt": 1, "gesamt": 4}


# ---------------------------------------------------------------- Formate
def test_formate_deutsch():
    assert V.de(1.234, 2) == "1,23" and V.de_vz(1.234, 1) == "+1,2" and V.de_vz(-0.04, 1) == "0,0" and V.de(-0.001, 2) == "0,00"
    assert V.zahl(246384) == "246 384" and V.faktor(162.1) == "162" and V.faktor(7.46) == "7,5"
    assert V.wiss(1.0e-11) == "1,0e-11" and V.wiss(1.865) == "1,865"
    assert V.intervall(-2.07, 0.4) == "[-2,1; +0,4]"


# ---------------------------------------------------------------- E0
def test_verdict_instanz():
    art, t = V.verdict_instanz("Zeigernetz", -3.14)
    assert art == "info" and "um 3,1 Prozentpunkte kürzer" in t and "sagt über die Zellen unten nichts" in t
    assert "länger" in V.verdict_instanz("Merkmalsnetz", 2.0)[1] and "Prozent der Länge" in V.verdict_instanz("Merkmalsnetz", 2.0, "Prozent der Länge des Nächsten Nachbarn")[1]


# ---------------------------------------------------------------- E1
def test_verdict_skala_bricht_warning_und_nennt_zahlen():
    erg = [{"name": "Zeigernetz (A)", "z": _z(115.91, 100.0, 130.0, k=5), "invariant": False}, {"name": "Merkmalsnetz (B mit s)", "z": _z(-9.1, -11.0, -7.0, k=0), "invariant": True}]
    art, t = V.verdict_skala(10.0, 0, 0, erg, 100, V.kosten_zeile(20, 190, 162.1, 31.08))
    assert art == "warning"
    for teil in ("Faktor 10,00", "Verschiebung (0; 0)", "Zeigernetz (A) bricht unter dem Nächsten Nachbarn", "+115,9", "[+100,0; +130,0]", "5 von 5 Zufalls-Seeds darüber", "100 Instanzen",
                 "schlägt den Nächsten Nachbarn", "folgt aus der Konstruktion", "K = 190 für alle drei gleich", "das 162-Fache", "das 31-Fache"):
        assert teil in t, teil


def test_verdict_skala_success_wenn_keiner_bricht_und_basis_text():
    erg = [{"name": "Zeigernetz (A+nt)", "z": _z(-8.0, -10.0, -6.0, k=0), "invariant": False}]
    art, t = V.verdict_skala(1.0, 0, 0, erg, 50)
    assert art == "success" and "Keine Veränderung: das ist die Basisinstanz." in t


def test_verdict_skala_nicht_unterscheidbar_und_kleine_stichprobe():
    erg = [{"name": "Zeigernetz (A)", "z": _z(0.5, -1.0, 2.0, k=3), "invariant": False}]
    art, t = V.verdict_skala(0.25, 3, 2, erg, 5)
    assert art == "info" and "Stichprobe klein" in t and "nicht unterscheidbar" in t
    kein = [{"name": "Zeigernetz (A)", "z": _z(1.0, None, None, k=1, m=1, klasse=None), "invariant": False}]
    assert "kein Intervall" in V.verdict_skala(2.0, 0, 0, kein, 1)[1]


# ---------------------------------------------------------------- E2 bis E5
def test_verdict_groesse_zaehlt_und_nennt_die_latte():
    a = {10: "schlaegt", 20: "schlaegt", 50: "haelt", 100: "bricht", 200: "bricht"}
    b = {10: "schlaegt", 20: "schlaegt", 50: "schlaegt", 100: "schlaegt", 200: "haelt"}
    art, t = V.verdict_groesse("gleichverteilt", "Zeigernetz (A)", "Merkmalsnetz (B mit s)", a, b, {10: 100, 20: 100, 50: 100, 100: 50, 200: 30}, 50, 5.6, 16.5, "Kostenzeile.")
    assert art == "warning"
    for teil in ("in 2 von 5 Größen (zuerst bei n = 100)", "bricht in 0 von 5 Größen, schlägt ihn in 4", "100, 100, 100, 50, 30", "Nächster Nachbar mit 2-opt liegt bei n = 50 bei 5,6 % Lücke gegen 16,5 %", "Kostenzeile."):
        assert teil in t, teil
    art2, t2 = V.verdict_groesse("Gitter", "A", "B", {10: "haelt", 20: "haelt"}, {10: "schlaegt", 20: "schlaegt"}, {10: 100, 20: 100}, 20, None, None)
    assert art2 == "success" and "in 0 von 2 Größen" in t2 and "2-opt" not in t2


def test_verdict_verteilung_normierung_aendert_das_urteil_nur_bei_verschiedenen_urteilen():
    ak = {"gleich": "bricht", "geclustert": "bricht", "Gitter": "haelt"}
    ank = {"gleich": "bricht", "geclustert": "haelt", "Gitter": "haelt"}
    bk = {"gleich": "schlaegt", "geclustert": "schlaegt", "Gitter": "schlaegt"}
    art, t = V.verdict_verteilung(50, "Zeigernetz (A)", "Merkmalsnetz", ak, bk, {"gleich": 1.0, "geclustert": 1.89, "Gitter": -1.0}, {"gleich": 1.0, "geclustert": 1.00, "Gitter": -1.0}, "", ank)
    assert art == "warning" and "bricht in 2 von 3 Familien (gleich, geclustert)" in t and "Merkmalsnetz in 0" in t and "ändert bei geclustert das Urteil" in t and "+1,89 gegen +1,00" in t
    _, t2 = V.verdict_verteilung(20, "A", "B", {"gleich": "haelt"}, {"gleich": "haelt"}, {"gleich": 0.0}, {"gleich": 0.0}, "", {"gleich": "haelt"})
    assert "in keiner der Familien" in t2


def test_verdict_stichprobe():
    art, t = V.verdict_stichprobe(10, 143, 200, 30, "bricht", 3.46, 200)
    assert art == "info" and "in 143 von 200 Teilstichproben" in t and "±3,5 Prozentpunkte" in t and "Bei n = 200 sind 30 Instanzen gemessen" in t
    assert "Teilstichprobe gleich Vollstichprobe" in V.verdict_stichprobe(30, 0, 0, 30, "bricht", 0.0, 200)[1]


def test_ursache_nur_mit_zaehlung_schwellen_gesetzt():
    assert C.MISCH_AM_TRAINING_BASIS == 1 and C.MISCH_AM_TRAINING_WECHSEL == 3 and C.MISCH_AN_EINGABE == 4
    assert "am Training bei einer einzigen Größe" in V.ursache_satz(1, 3)
    assert "in dieser Rezeptur gegen" in V.ursache_satz(4, 0) and "in dieser Rezeptur gegen" in V.ursache_satz(6, 18) and "andere Rezepte" in V.ursache_satz(5, 18) and "nicht gemessen" in V.ursache_satz(5, 18)
    assert "nicht entschieden" in V.ursache_satz(2, 0) and "nicht entschieden" in V.ursache_satz(1, 4) and "nicht entschieden" in V.ursache_satz(3, 3)
    assert "nicht entschieden" in V.ursache_satz(1, 0, aaug_schwach=True)             # ein schwaches Aaug belegt „am Training“ nie
    assert "in dieser Rezeptur gegen" in V.ursache_satz(5, 18, aaug_schwach=True)


def test_verdict_mischtraining_nennt_alle_zahlen_und_die_modellguete_marke():
    art, t = V.verdict_mischtraining(5, 7, 2, 6, 4, 18, 1, 3, V.ursache_satz(2, 4), aaug_seeds=0)
    for teil in ("in 5 von 15 Zellen", "ohne Mischtraining in 7", "ab n = 100 2 von 6", "mit Augmentierung bricht es in 4 von 18 Zellen", "bricht es in 1 von 3 Zellen", "nicht entschieden",
                 "ist schwach: nur 0 von 5 Zufalls-Seeds", "nichts über Augmentierung im Allgemeinen"):
        assert teil in t, teil
    assert art == "info"
    assert "schwach" not in V.verdict_mischtraining(5, 7, 2, 6, 4, 18, 1, 3, "x", aaug_seeds=5)[1] and "schwach" not in V.verdict_mischtraining(5, 7, 2, 6, 4, 18, 1, 3, "x")[1]


def test_kosten_zeile_nennt_k_und_faktoren():
    t = V.kosten_zeile(50, 1225, 311.0, 48.7)
    assert "K = 1 225" in t and "das 311-Fache" in t and "das 49-Fache" in t and "n = 50" in t

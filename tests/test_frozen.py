"""Eingefrorene Messung (`nfi_frozen.json`): Schema, endliche Werte, Instanzzahlen, Referenz höchstens gleich jeder gespeicherten Länge, Neu-Rechnen eines Ausschnitts als Band
(nie bytegleich: Linux rechnet BLAS-Summen und Gleichstände anders), Invarianz der Netze unter den Wechseln, die Zählungen nur aus den gespeicherten Werten."""

import json
from pathlib import Path

import numpy as np
import pytest

import nfi_constants as C
import nfi_eval as EV
import nfi_exact as EXA
import nfi_experiments as EX
import nfi_models as MO
import nfi_transform as TR

ROOT = Path(__file__).resolve().parent.parent
PFAD = ROOT / "nfi_frozen.json"
pytestmark = pytest.mark.skipif(not PFAD.exists(), reason="nfi_frozen.json fehlt")

FR = EX.Frozen.laden() if PFAD.exists() else None
ROH = json.loads(PFAD.read_text(encoding="utf-8")) if PFAD.exists() else None


def test_format_und_netze():
    assert ROH["format"] == "nfi-frozen-1" and ROH["meta"]["seeds"] == C.NETZ_SEEDS
    assert {"A", "Ant", "An", "Bmit", "Bohne"} <= set(ROH["meta"]["netze"])
    assert ROH["meta"]["s0"] == C.S0 and ROH["instanzen"] == {str(n): C.INSTANZEN[n] for n in C.GROESSEN}


@pytest.mark.parametrize("n", C.GROESSEN)
@pytest.mark.parametrize("fam", C.FAMILIEN)
def test_zelle_hat_die_angegebene_zahl_instanzen_und_endliche_werte(fam, n):
    z = FR.basis(fam, n)
    m = C.INSTANZEN[n]
    assert len(z["ref"]) == len(z["nn"]) == m
    for netz, L in z["modelle"].items():
        L = np.array(L)
        assert L.shape == (C.NETZ_SEEDS, m) and np.isfinite(L).all() and (L > 0).all(), netz
    ref, nn = FR.ref(fam, n), FR.nn(fam, n)
    assert np.isfinite(ref).all() and (nn >= ref - 1e-6).all()
    for netz in z["modelle"]:
        assert (FR.laengen(fam, n, netz) >= ref[None, :] - 1e-6).all(), (fam, n, netz)       # Referenz höchstens gleich jeder gespeicherten Länge (bis n = 20 exakt, sonst beste bekannte)
    if z.get("nn2opt"):
        assert (np.array(z["nn2opt"]) >= ref[:len(z["nn2opt"])] - 1e-6).all() and len(z["nn2opt"]) == (m if n < 200 else C.NN2OPT_INSTANZEN_GROSS)


@pytest.mark.parametrize("n", C.WECHSEL_GROESSEN)
@pytest.mark.parametrize("fam", C.FAMILIEN)
def test_wechselzellen_haben_a_und_b_ohne_s_und_die_invarianten_netze_gleiche_touren(fam, n):
    for art in ("klein", "gross", "versch"):
        w = ROH["wechsel"][f"{fam}_{n}_{art}"]
        assert {"A", "Bohne"} <= set(w["modelle"]), (fam, n, art)
        for netz, L in w["modelle"].items():
            L = np.array(L)
            assert L.shape == (C.NETZ_SEEDS, C.INSTANZEN[n]) and np.isfinite(L).all() and (L >= FR.ref(fam, n)[None, :] - 1e-6).all()
        assert w["invarianz"] and min(w["invarianz"].values()) >= 0.99, w["invarianz"]


def test_a_roh_bricht_bei_faktor_10_in_den_gleichverteilten_zellen():
    for n in C.WECHSEL_GROESSEN:
        z = FR.zelle("gleich", n, "A", "gross")
        assert z["klasse"] == "bricht" and z["luecke"] > FR.luecke_nn("gleich", n) + 50.0     # mindestens 50 Prozentpunkte über dem Nächsten Nachbarn


def test_neu_rechnen_eines_ausschnitts_als_band():
    """Drei Instanzen je Zelle für n = 20 und 50 (Basis und die drei Wechsel), Zellenmittel der Länge gegen die gespeicherten Werte: höchstens 0,5 Prozent (nicht bytegleich)."""
    for n in C.WECHSEL_GROESSEN:
        for fam in C.FAMILIEN:
            X = TR.testmenge(fam, n, 3)
            D = TR.abstaende(X)
            for art in C.WECHSEL:
                netze = ("A", "Ant", "An", "Bmit", "Bohne") if art == "basis" else ("A", "Bohne")
                for netz in netze:
                    L, _ = EV.laengen_netz(netz, MO.lade_alle(netz, [0, 1]), TR.wechsel(X, art), D)
                    alt = FR.laengen(fam, n, netz, art)[:2, :3]
                    assert abs(100 * (L.mean() - alt.mean()) / alt.mean()) <= 0.5, (fam, n, art, netz)


def test_referenzen_bis_n_20_sind_exakt():
    for fam in C.FAMILIEN:
        for n in (10, 20):
            X = TR.testmenge(fam, n, 2)
            D = TR.abstaende(X)
            assert FR.ref(fam, n)[:2] == pytest.approx([EXA.held_karp(D[0]), EXA.held_karp(D[1])], abs=2e-7)


def test_die_zaehlungen_kommen_nur_aus_den_gespeicherten_werten():
    import nfi_befunde as BF
    b = BF.befunde(FR)
    k, s, g = b["wechsel_A"]
    assert g == 18 and 0 <= k <= 18
    assert b["basis_A"][2] == 15
    for netz in ("Ant", "An", "Bmit"):                  # Gleichheit mit der Basiszelle: ein Wechselbruch gibt es nur, wo die Basiszelle (n = 20, 50) selbst bricht, dann in allen drei Wechseln
        bricht_basis = sum(1 for (f, n), v in FR.basis_klassen(netz).items() if n in C.WECHSEL_GROESSEN and v == "bricht")
        assert b[f"wechsel_{netz}"][0] == 3 * bricht_basis, netz

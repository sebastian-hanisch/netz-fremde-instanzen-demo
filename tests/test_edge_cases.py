"""Randfälle (DEMO-PLAYBOOK §12): kleinste Größen, eine Instanz, ein Seed, Intervall nicht berechenbar, Spannweite null, alle Regler am Rand, Gleichstand, Rechengrenzen."""

import numpy as np
import pytest

import nfi_constants as C
import nfi_eval as EV
import nfi_evaluation as V
import nfi_experiments as EX
import nfi_hand as HA
import nfi_instances as I
import nfi_kosten as KO
import nfi_models as MO
import nfi_transform as TR


@pytest.mark.parametrize("n", [4, 5, 10])
def test_kleine_groessen_alle_netze_liefern_touren(n):
    X = TR.testmenge("gleich", n, 3)
    D = TR.abstaende(X)
    for netz in ("A", "Ant", "Bmit", "Bohne"):
        L, T = EV.laengen_netz(netz, [MO.lade(netz, 0)[0]], X, D)
        assert T.shape == (1, 3, n) and TR.ist_permutation(T[0]) and np.isfinite(L).all()


def test_reglergrenzen_faktor_und_verschiebung_am_rand():
    X = TR.testmenge("cluster", 20, 3)
    D = TR.abstaende(X)
    for faktor in (C.FAKTOR_STUFEN[0], C.FAKTOR_STUFEN[-1]):
        for vx, vy in ((C.VERSCHIEBUNG_MIN, C.VERSCHIEBUNG_MIN), (C.VERSCHIEBUNG_MAX, C.VERSCHIEBUNG_MAX)):
            Xw = TR.wechsel_frei(X, faktor, vx, vy)
            for netz in ("A", "Bohne"):
                L, T = EV.laengen_netz(netz, [MO.lade(netz, 0)[0]], Xw, D)
                assert TR.ist_permutation(T[0]) and np.isfinite(L).all()


def test_ein_seed_und_eine_instanz_geben_kein_intervall_und_kein_urteil():
    L = np.array([[3.0], [3.1]])
    z = EV.zelle(L, np.array([3.5]), np.array([3.0]))
    assert z["lo"] is None and z["klasse"] is None and z["m"] == 1 and z["seeds"] == 2
    z2 = EV.zelle(np.array([[3.2, 3.3, 3.4]]), np.array([3.5, 3.5, 3.5]), np.array([3.0, 3.0, 3.0]))
    assert z2["seeds"] == 1 and z2["klasse"] == "schlaegt" and z2["v"] < 0


def test_alle_orte_gleich_spannweite_null_gibt_nullen_statt_division_durch_null():
    X = np.full((2, 6, 2), 0.7)
    assert np.array_equal(I.normalize_unit(X), np.zeros((2, 6, 2)))


def test_gleichstand_der_zeiger_logits_waehlt_den_kleinsten_index():
    u = HA.zeiger_logits_spielzeug(10.0 * HA.PUNKTE, 1, [2, 3, 4])
    assert HA.spannweite(u) < 1e-9 and int(np.argmax(u)) == 0


def test_nenner_null_in_verdikten_alle_halten_kein_bricht_ab_n():
    a = {n: "haelt" for n in C.GROESSEN}
    _, t = V.verdict_groesse("gleichverteilt", "A", "B", a, a, {n: 10 for n in C.GROESSEN}, 50, None, None)
    assert "in 0 von 5 Größen" in t and "zuerst bei" not in t


def test_rechengrenzen_cap_live_und_e4():
    assert KO.cap_live(50, ("A", "Bohne"), 5, 100) <= C.LIVE_M_MAX
    for n in (10, 20, 50):
        m = KO.cap_live(n, ("A", "Bohne"), 5, C.LIVE_M_MAX)
        assert 1 <= m <= C.LIVE_M_MAX and (KO.live_o(n, ("A", "Bohne"), 5, m) <= C.LIVE_O_MAX or m == 1)
    assert KO.cap_live(200, ("A", "Bohne"), 5, 99, m_max=6) == 6 and KO.live_o(200, ("A", "Bohne"), 5, 6) <= C.LIVE_O_MAX
    assert EX.e4_cap(100, 300) * C.E4_BOOT * 100 <= C.E4_ZIEHUNGEN_MAX and EX.e4_cap(3, 300) == 300 and EX.e4_cap(10, 5) == 5


def test_teilstichproben_sind_reproduzierbar_und_urteilen_nach_der_regel():
    delta = np.vstack([np.linspace(3.0, 7.0, 40)] * 5)          # alle Instanzen deutlich über null, alle fünf Seeds
    r1 = EX.teilstichproben(delta, 10, 50, schluessel=1)
    r2 = EX.teilstichproben(delta, 10, 50, schluessel=1)
    assert r1["klassen"] == r2["klassen"] == ["bricht"] * 50 and np.array_equal(r1["v"], r2["v"])
    assert r1["m"] == 10 and r1["breite"] > 0
    neg = EX.teilstichproben(-delta, 10, 20)
    assert set(neg["klassen"]) == {"schlaegt"}
    null = EX.teilstichproben(np.zeros((5, 40)), 6, 30)
    assert set(null["klassen"]) == {"haelt"}                      # kein Unterschied: nie „bricht“ oder „schlägt“

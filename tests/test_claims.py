"""Jede Zahl der README: Zeile für Zeile gegen die eingefrorene Messung, als Band (Linux rechnet das Neu-Rechnen im letzten Bit anders) oder exakt (Zählungen, Formeln, Konstanten).

Die README führt in „Befunde (gemessen, keine Behauptungen)“ die Spalte „Test“; die Namen stehen hier. Zahlen, die nur in der README stehen (Vormessung, Literatur), sind dort als
„Messung dieser README, nicht im Test“ gekennzeichnet."""

from pathlib import Path

import numpy as np
import pytest

import nfi_befunde as BF
import nfi_constants as C
import nfi_evaluation as V
import nfi_experiments as EX
import nfi_kosten as KO

ROOT = Path(__file__).resolve().parent.parent
PFAD = ROOT / "nfi_frozen.json"
pytestmark = pytest.mark.skipif(not PFAD.exists(), reason="nfi_frozen.json fehlt")

FR = EX.Frozen.laden() if PFAD.exists() else None
B = BF.befunde(FR) if PFAD.exists() else None
README = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""


def test_luecken_n20():
    assert 14.0 <= B["nn_luecke_20"] <= 19.0 and abs(B["nn_luecke_20"] - 16.48) < 0.01
    for netz in ("A", "Ant", "An"):
        assert 8.0 <= B[f"luecke_{netz}_20"] <= 13.0
        assert FR.zelle("gleich", 20, netz)["v"] < -3.0                                  # Unterschied zum Nächsten Nachbarn unter −3 Prozentpunkten
    for netz in ("Bmit", "Bohne"):
        assert 6.0 <= B[f"luecke_{netz}_20"] <= 11.0


def test_wechselzaehlung():
    assert B["wechsel_A"][0] == 14 and B["wechsel_A"][2] == 18
    for netz in ("Ant", "An", "Bmit"):
        assert B[f"wechsel_{netz}"][0] == 0                                              # Konstruktion: gleiche Eingabe wie in der Basiszelle
    assert B["wechsel_Bohne"][0] == 4
    assert B["wechsel_luecke_ueber_basis_A"] == 18                                       # in keiner der 18 Zellen bleibt A bei der Lücke der Basiszelle
    gross = [FR.zelle(f, n, "Bohne", "gross")["klasse"] for n in C.WECHSEL_GROESSEN for f in C.FAMILIEN]
    rest = [FR.zelle(f, n, "Bohne", a)["klasse"] for n in C.WECHSEL_GROESSEN for f in C.FAMILIEN for a in ("klein", "versch")]
    assert gross.count("bricht") == 4 and len(gross) == 6 and rest.count("bricht") == 0 and len(rest) == 12
    assert 12 <= B["wechsel_A"][0] <= 16 and 2 <= B["wechsel_Bohne"][0] <= 6              # Bänder gegen Streuung
    assert B["invarianz_min"] >= 0.99


def test_a_faktor_10():
    assert 90 <= B["A_gross_luecke_20"] <= 200 and abs(B["A_gross_luecke_20"] - 136.86) < 0.01
    assert 200 <= B["A_gross_luecke_50"] <= 350 and abs(B["A_gross_luecke_50"] - 281.99) < 0.01
    assert B["A_gross_luecke_20"] > B["nn_luecke_20"] + 50 and B["A_gross_luecke_50"] > B["nn_luecke_50"] + 50
    assert abs(B["nn_luecke_50"] - 20.90) < 0.01


def test_basiszaehlung():
    assert B["basis_A"][0] == 6 and B["basis_Ant"][0] == 5 and B["basis_An"][0] == 5 and B["basis_Bmit"][0] == 0 and B["basis_Bohne"][0] == 0
    assert 4 <= B["basis_A"][0] <= 8 and 3 <= B["basis_Ant"][0] <= 7 and 3 <= B["basis_An"][0] <= 7
    assert B["gross_A"] == (5, 0, 6) or tuple(B["gross_A"]) == (5, 0, 6)
    assert B["gross_Ant"][0] == 5 and B["gross_An"][0] == 5
    assert B["A_erstes_n_gleich"] == 100 and B["A_erstes_n_cluster"] == 50
    assert B["basis_Bmit"][1] == 15 and B["basis_Bohne"][1] == 15 and B["basis_Bmit"][2] == 15
    z = FR.zelle("cluster", 50, "Ant")
    assert z["klasse"] == "haelt" and V.nicht_unterscheidbar(z["lo"], z["hi"]) and abs(z["v"] - 0.78) < 0.01
    z = FR.zelle("cluster", 50, "An")
    assert z["klasse"] == "haelt" and abs(z["v"] - 0.81) < 0.01
    z = FR.zelle("gitter", 100, "A")
    assert z["klasse"] == "haelt" and abs(z["v"] - (-0.47)) < 0.01


def test_unterschiede_b():
    assert -6.0 <= B["Bmit_unterschied_100"] <= -1.5 and abs(B["Bmit_unterschied_100"] - (-3.04)) < 0.01
    assert -7.0 <= B["Bmit_unterschied_200"] <= -1.0 and abs(B["Bmit_unterschied_200"] - (-3.74)) < 0.01
    z = FR.zelle("gleich", 100, "Bmit")
    assert abs(z["lo"] - (-4.49)) < 0.02 and abs(z["hi"] - (-1.64)) < 0.02
    assert abs(FR.zelle("gleich", 100, "Bohne")["v"] - (-8.95)) < 0.01 and abs(FR.zelle("gleich", 200, "Bohne")["v"] - (-6.77)) < 0.01
    assert FR.zelle("gleich", 50, "Bohne")["luecke"] < FR.zelle("gleich", 50, "Bmit")["luecke"] and FR.zelle("gleich", 100, "Bohne")["luecke"] < FR.zelle("gleich", 100, "Bmit")["luecke"]
    assert abs(FR.zelle("gleich", 50, "Bohne")["luecke"] - 11.35) < 0.01 and abs(FR.zelle("gleich", 50, "Bmit")["luecke"] - 16.29) < 0.01


def test_normierung_kostet():
    assert B["norm_kosten_max"] <= 4.0 and abs(B["norm_kosten_max"] - 2.20) < 0.01


def test_diagnosen():
    s = B["saettigung"]
    assert s["gross"] >= 0.90 and s["klein"] <= 0.10 and s["versch"] > s["basis"]
    assert abs(100 * s["basis"] - 47.6) < 0.1 and abs(100 * s["klein"] - 0.1) < 0.1 and abs(100 * s["gross"] - 99.0) < 0.1 and abs(100 * s["versch"] - 80.7) < 0.1
    k = B["kappung_x1"]
    assert 0.10 <= k["basis"] <= 0.35 and k["gross"] >= 0.90 and abs(100 * k["basis"] - 20.7) < 0.1 and abs(100 * k["gross"] - 98.5) < 0.1
    for n, soll in zip(C.GROESSEN, (1.42, 1.00, 0.60, 0.43, 0.30)):
        assert abs(B["s_verhaeltnis"][n] - soll) <= 0.15 and abs(B["s_verhaeltnis"][n] - soll) < 0.01


def test_luecken_n50():
    g2 = FR.luecke_nn2opt("gleich", 50)
    assert 3.0 <= g2 <= 7.0 and abs(g2 - 5.00) < 0.01
    assert g2 < FR.zelle("gleich", 50, "Bohne")["luecke"] < FR.zelle("gleich", 50, "Bmit")["luecke"] < FR.zelle("gleich", 50, "A")["luecke"]
    assert abs(FR.zelle("gleich", 50, "A")["luecke"] - 19.85) < 0.01 and abs(FR.zelle("gleich", 50, "Ant")["luecke"] - 19.62) < 0.01


def test_teilstichproben():
    z = FR.zelle("gleich", 100, "Bmit")
    d = FR.delta_je_seed("gleich", 100, "Bmit")
    res = EX.teilstichproben(d, 10, EX.e4_cap(10, 200), schluessel=100 * 10 + 0)
    gleich = sum(1 for k in res["klassen"] if k == z["klasse"])
    assert z["klasse"] == "schlaegt" and 40 <= gleich <= 120 and gleich == 69
    za = FR.zelle("gleich", 200, "A")
    ra = EX.teilstichproben(FR.delta_je_seed("gleich", 200, "A"), 10, EX.e4_cap(10, 200), schluessel=200 * 10 + 0)
    assert za["klasse"] == "bricht" and sum(1 for k in ra["klassen"] if k == "bricht") >= 190 and abs(ra["breite"] - 3.2) < 0.1


def test_referenz_unterboten():
    for k, v in B["unterboten"].items():
        assert v <= 0.05, k


def test_kosten_exakt():
    assert {n: KO.k_konstruktion(n) for n in C.GROESSEN} == {10: 45, 20: 190, 50: 1225, 100: 4950, 200: 19900}
    assert [KO.ops_nn(n) for n in C.GROESSEN] == [360, 1520, 9800, 39600, 159200]
    assert [KO.ops_zeiger(n) for n in C.GROESSEN] == [113704, 246384, 764424, 2027824, 6054624]
    assert [KO.ops_merkmal(n) for n in C.GROESSEN] == [9298, 47242, 454827, 2837774, 19408225]
    assert KO.ops_merkmal(20, True) == 47642
    assert round(KO.kosten_je_konstruktion(20)["f_zeiger"]) == 162 and round(KO.kosten_je_konstruktion(20)["f_merkmal"]) == 31
    assert (C.ZEIGER_PARAMS, C.MERKMAL_PARAMS, C.TOUREN_ZEIGER, C.TOUREN_MERKMAL, C.ZEILEN_MERKMAL) == (12480, 145, 960000, 960000, 182400000)
    assert C.INSTANZEN == {10: 100, 20: 100, 50: 100, 100: 100, 200: 50} and C.S0 == 0.1239


def test_s0_ist_das_mittlere_s_der_trainingsinstanzen():
    import nfi_instances as I
    import nfi_scenario as SC
    import nfi_transform as TR
    from nfi_rngvec import UStream
    X = I.train_batch(UStream(SC.derive(C.ZWECK_TRAIN_X, 20, 99)), 3000, 20)
    assert abs(float(TR.skala_s(I.distance_matrix_np(X)).mean()) - C.S0) < 0.002


def test_readme_nennt_die_zahlen_und_die_testzahl():
    if not README:
        pytest.skip("README fehlt")
    for teil in ("14 von 18", "136,86", "281,99", "2,20", "47,6", "99,0", "98,5", "69 von 200", "190", "246 384", "47 242", "12 480", "7,97"):
        assert teil in README, teil


def test_preset_verdikte_je_preset():
    """Je Preset die erwartete Verdikt-Zeile aus den Live-Zellen (12 Instanzen, fünf Zufalls-Seeds; DEMO-PLAYBOOK §11a Punkt 7)."""
    import nfi_models as MO
    if not MO.vorhanden("A"):
        pytest.skip("Modelle fehlen")
    erwartet = {   # Preset -> (Wechsel, Zeigernetz, Merkmalsnetz, Urteil Zeigernetz, Urteil Merkmalsnetz)
        "Faktor 10, Zeigernetz roh": ("gross", "A", "Bmit", "bricht", "schlaegt"),
        "Faktor 10, Eingabe normiert": ("gross", "Ant", "Bmit", "schlaegt", "schlaegt"),
        "Merkmalsnetz ohne s, Faktor 10": ("gross", "Ant", "Bohne", "schlaegt", "bricht"),
    }
    for name, (art, z_netz, m_netz, u_z, u_m) in erwartet.items():
        assert name in C.PRESETS
        zz = EX.live_zelle("gleich", 20, z_netz, list(range(12)), 5, art, FR)
        zm = EX.live_zelle("gleich", 20, m_netz, list(range(12)), 5, art, FR)
        assert zz["klasse"] == u_z and zm["klasse"] == u_m, (name, zz["klasse"], zm["klasse"])
        assert np.isfinite(zz["v"]) and np.isfinite(zm["v"])


def test_mischtraining_e5():
    assert FR.seeds_unter_nn("Aaug") == 0 and FR.seeds_unter_nn("Amix") == 5 and all(FR.seeds_unter_nn(n) == 5 for n in ("A", "Ant", "An", "Bmit", "Bohne"))
    assert abs(B["luecke_Aaug_20"] - 82.69) < 0.01 and abs(B["luecke_Amix_20"] - 11.05) < 0.01
    assert tuple(B["basis_Aaug"]) == (15, 0, 15) and tuple(B["gross_Aaug"]) == (6, 0, 6) and tuple(B["wechsel_Aaug"]) == (18, 0, 18)
    assert B["basis_Amix"][0] == 5 and B["basis_Amix"][1] == 9 and B["gross_Amix"][0] == 5 and B["gross_Amix"][1] == 1 and B["wechsel_Amix"][0] == 0
    assert B["basis_Ant"][1] == 7
    k200 = V.zaehle_klassen(v for (f, n), v in FR.basis_klassen("Amix").items() if n == 200)
    assert k200["bricht"] == 3 and k200["gesamt"] == 3
    assert abs(FR.zelle("gleich", 50, "Amix")["v"] - (-1.98)) < 0.01 and FR.zelle("gleich", 50, "Amix")["klasse"] == "schlaegt" and FR.zelle("gleich", 50, "Ant")["klasse"] == "haelt"
    assert abs(FR.zelle("gitter", 100, "Amix")["v"] - (-2.08)) < 0.01 and FR.zelle("gitter", 100, "Amix")["klasse"] == "schlaegt"
    assert abs(FR.zelle("gleich", 200, "Amix")["v"] - 11.90) < 0.01 and abs(FR.zelle("gleich", 200, "Ant")["v"] - 12.67) < 0.01
    assert "spricht in dieser Rezeptur gegen" in V.ursache_satz(5, 18, True)       # Zählung Amix ab n = 100: 5 von 6, Aaug schwach

"""Handbeispiel von Hand (`neuro-kombinatorik-planung/handbeispiele_05.py`, 87 Prüfungen) als Tests, mit denselben Toleranzen: fünf Orte, Nächster Nachbar, Optimum, Skala und Lage,
Spielzeug-Zeigerstufe, die sieben Merkmale, Parameterzahlen, Kosten und die Urteilsregel. Die Vektorfassungen der App werden gegen die Schleifenfassung in `nfi_hand.py` gehalten."""

import math

import numpy as np
import pytest
from conftest import HAND, HAND_NN_TOUR, HAND_OPT_LAENGE

import nfi_constants as C
import nfi_cost as K0
import nfi_evaluation as V
import nfi_exact as EXA
import nfi_gegner as GG
import nfi_hand as HA
import nfi_instances as I
import nfi_kosten as KO
import nfi_net as N
import nfi_transform as TR
import nfik_featnet as KF
import nfik_mlp as MLP
from nfi_rngvec import UStream

S0 = C.S0


# ---------------------------------------------------------------- H1 Nächster Nachbar, Optimum, Zählung
def test_h1_abstaende_von_hand():
    D = HA.abstaende(HAND)
    assert D[0, 1] == pytest.approx(math.sqrt(0.2)) and D[1, 2] == pytest.approx(0.6) and D[2, 4] == pytest.approx(math.sqrt(0.8))
    assert D[0, 2] == pytest.approx(math.sqrt(1.04)) and D[0, 3] == pytest.approx(math.sqrt(1.28))


def test_h1_nachster_nachbar_tour_laenge_und_zaehlung():
    D = HA.abstaende(HAND)
    t, k = HA.nn_tour_gezaehlt(D)
    assert t == HAND_NN_TOUR and k == 10 == KO.k_konstruktion(5) == K0.k_konstruktion(5) == 5 * 4 // 2
    L = HA.tour_laenge(D, t)
    assert L == pytest.approx(math.sqrt(0.2) + math.sqrt(0.2) + math.sqrt(0.4) + math.sqrt(0.4) + math.sqrt(1.04))
    assert abs(L - 3.1791) < 5e-5
    # Vektorfassung und die Kopie aus Stück 1 liefern dieselbe Tour, K gleich n(n-1)/2
    assert TR.nn_touren(D[None])[0].tolist() == HAND_NN_TOUR
    t2, k2 = GG.nn_tour_k(D.tolist(), 0)
    assert t2 == HAND_NN_TOUR and k2 == 10


def test_h1_optimum_aufzaehlung_held_karp_und_luecke():
    D = HA.abstaende(HAND)
    L_opt, t_opt, n_touren, n_verschieden = HA.optimum_aufzaehlung(D)
    assert n_touren == math.factorial(4) == 24 and n_verschieden == 12
    assert L_opt == pytest.approx(HAND_OPT_LAENGE) and abs(L_opt - 2.9446) < 5e-5
    assert EXA.held_karp(D) == pytest.approx(L_opt)
    assert HA.tour_laenge(D, EXA.held_karp_tour(D)[1]) == pytest.approx(L_opt)
    assert t_opt in ([0, 4, 3, 2, 1], [0, 1, 2, 3, 4])
    L_nn = HA.tour_laenge(D, HAND_NN_TOUR)
    assert abs(100 * (L_nn - L_opt) / L_opt - 7.97) < 5e-3


# ---------------------------------------------------------------- H2 Skala und Lage
@pytest.mark.parametrize("art", ["klein", "gross", "versch"])
def test_h2_normierung_nn_und_d_durch_s_sind_invariant(art):
    D = HA.abstaende(HAND)
    Xt = TR.wechsel(HAND, art)
    Dt = HA.abstaende(Xt)
    assert HA.nn_tour_gezaehlt(Dt)[0] == HAND_NN_TOUR
    assert np.allclose(I.normalize_unit(Xt), I.normalize_unit(HAND), atol=1e-12)
    fak = {"klein": 0.25, "gross": 10.0, "versch": 1.0}[art]
    assert HA.tour_laenge(Dt, HAND_NN_TOUR) == pytest.approx(fak * HA.tour_laenge(D, HAND_NN_TOUR))
    assert np.allclose(Dt / HA.skala_s(Dt), D / HA.skala_s(D), atol=1e-12)


def test_h2_normierung_wertebereich_und_seitenverhaeltnis():
    Pn = I.normalize_unit(HAND)
    assert Pn.max() == pytest.approx(1.0) and Pn.min() == pytest.approx(0.0)
    Pw = HAND * np.array([1.0, 0.5])
    n2 = I.normalize_unit(Pw)
    assert n2[:, 0].max() == pytest.approx(1.0) and n2[:, 1].max() == pytest.approx(0.4)


# ---------------------------------------------------------------- H3 Spielzeug-Zeigerstufe
def test_h3_spielzeug_logits_und_saettigung():
    frei = [2, 3, 4]
    u = HA.zeiger_logits_spielzeug(HAND, 1, frei)
    assert u == pytest.approx([10 * math.tanh(0.44 / math.sqrt(2)), 10 * math.tanh(0.48 / math.sqrt(2)), 10 * math.tanh(0.20 / math.sqrt(2))])
    assert abs(u[0] - 3.015) < 5e-3 and abs(u[1] - 3.269) < 5e-3 and abs(u[2] - 1.405) < 5e-3
    sp = HA.spannweite(u)
    ug = HA.zeiger_logits_spielzeug(10.0 * HAND, 1, frei)
    assert sp > 1.8 and HA.spannweite(ug) < 1e-9 and int(np.argmax(ug)) == 0
    uk = HA.zeiger_logits_spielzeug(0.25 * HAND, 1, frei)
    assert abs(HA.spannweite(uk) - sp / 16) < 0.06
    assert HA.spannweite(HA.zeiger_logits_spielzeug(HAND + np.array([3.0, 2.0]), 1, frei)) < 1e-6
    un = HA.zeiger_logits_spielzeug(I.normalize_unit(10.0 * HAND + np.array([3.0, 2.0])), 1, frei)
    assert np.allclose(un, u, atol=1e-12)


# ---------------------------------------------------------------- H4 Merkmale
def test_h4_skala_und_merkmale_mit_s_von_hand():
    D = HA.abstaende(HAND)
    s = HA.skala_s(D)
    assert s == pytest.approx((math.sqrt(0.2) + math.sqrt(0.2) + 0.6 + math.sqrt(0.4) + math.sqrt(0.2)) / 5)
    F = HA.merkmale(D, s, cur=1, start=0, frei=[2, 3, 4], n_besucht=2)
    soll = {2: [1.1655, 0.125, 1.9810, 1.2286, 0.25, 0.4, 0.3416], 3: [1.4007, 0.25, 2.1977, 1.2286, 0.25, 0.4, 0.6125], 4: [0.8687, 0.0, 1.2286, 1.2286, 0.25, 0.4, 0.0]}
    for j in (2, 3, 4):
        assert np.allclose(F[j], soll[j], atol=6e-4), (j, F[j])


def test_h4_vektorfassung_der_merkmale_gleicht_der_schleife():
    D = HA.abstaende(HAND)
    s = HA.skala_s(D)
    Dd = D.copy()
    np.fill_diagonal(Dd, np.inf)
    unv = np.zeros((1, 5), dtype=bool)
    unv[0, [2, 3, 4]] = True
    X = KF.k_features(Dd[None], np.array([s]), np.array([1]), np.array([0]), unv, 2, 5)[0]
    F = HA.merkmale(D, s, 1, 0, [2, 3, 4], 2)
    for j in (2, 3, 4):
        assert np.allclose(X[j], F[j], atol=1e-12)


def test_h4_mit_s_alle_merkmale_unter_wechseln_gleich_ohne_s_nicht():
    D = HA.abstaende(HAND)
    F0 = HA.merkmale(D, HA.skala_s(D), 1, 0, [2, 3, 4], 2)
    for art in ("klein", "gross", "versch"):
        Dt = HA.abstaende(TR.wechsel(HAND, art))
        Ft = HA.merkmale(Dt, HA.skala_s(Dt), 1, 0, [2, 3, 4], 2)
        assert all(np.allclose(Ft[j], F0[j], atol=1e-12) for j in F0), art
    Fg = HA.merkmale(HA.abstaende(10.0 * HAND), S0, 1, 0, [2, 3, 4], 2)
    Fb = HA.merkmale(D, S0, 1, 0, [2, 3, 4], 2)
    for j in (2, 3, 4):
        assert Fg[j][0] == 6.0 and Fg[j][2] == 6.0 and Fg[j][3] == 6.0 and Fg[j][4] == 0.0       # x1, x3, x4 gekappt, x5 gleich 0
        assert Fg[j][1] == pytest.approx(F0[j][1]) and Fg[j][5] == pytest.approx(F0[j][5]) and Fg[j][6] == pytest.approx(F0[j][6])   # x2, x6, x7 unverändert
    assert Fb[2][2] == 6.0                                                                          # schon die Basisinstanz kappt x3 mit der festen Skala (n = 5 ist dünner als das Training)
    Fv = HA.merkmale(HA.abstaende(HAND + np.array([3.0, 2.0])), S0, 1, 0, [2, 3, 4], 2)
    assert all(np.allclose(Fv[j], Fb[j], atol=1e-12) for j in Fv)                                   # Verschiebung ändert auch ohne s nichts


# ---------------------------------------------------------------- H5 Parameter und Kosten
def test_h5_parameterzahlen():
    assert N.n_params(N.init_params(32, UStream(1), layers=1)) == 12480 == C.ZEIGER_PARAMS
    assert N.n_params(N.init_params(16, UStream(2), layers=1)) == 3168
    assert MLP.n_params(MLP.init_mlp(7, 16, UStream(3))) == 145 == C.MERKMAL_PARAMS


class _Zaehler:
    def __init__(self):
        self.n = 0

    def mm(self, a, b):
        self.n += a.shape[0] * a.shape[1] * b.shape[1]
        return a @ b


def _encoder_gezaehlt(n, d):
    z = _Zaehler()
    rng = np.random.default_rng(1)        # nur für die Zählung (Werte egal); die App benutzt SplitMix64
    X = rng.random((n, 2))

    def W(a, b):
        return rng.random((a, b))

    H = z.mm(X, W(2, d))
    Q, Kk, V_ = z.mm(H, W(d, d)), z.mm(H, W(d, d)), z.mm(H, W(d, d))
    S = z.mm(Q, Kk.T)
    AV = z.mm(S, V_)
    H1 = H + z.mm(AV, W(d, d))
    R = z.mm(H1, W(d, 2 * d))
    z.mm(R, W(2 * d, d))
    z.mm(H1, W(d, d))
    return z.n


def _decoder_gezaehlt(n, d):
    z = _Zaehler()
    rng = np.random.default_rng(2)
    row, Wd, Kt = rng.random((1, d)), rng.random((d, d)), rng.random((d, n))
    z.mm(row, Wd)
    z.mm(row, Wd)
    for _ in range(n - 1):
        c = z.mm(row, Wd)
        z.mm(c, Kt)
    return z.n + 4 * n * (n - 1)


def test_h5_gezaehlte_matrixprodukte_gleichen_der_formel():
    assert _encoder_gezaehlt(4, 2) == K0.ops_kodierer(4, 2) == 224 == 2 * 4 * 2 + 8 * 4 * 4 + 2 * 16 * 2 + 4 * 4
    for n, d in ((5, 3), (10, 8), (20, 32)):
        assert _encoder_gezaehlt(n, d) == K0.ops_kodierer(n, d)
    for n, d in ((4, 2), (5, 3), (20, 32)):
        assert _decoder_gezaehlt(n, d) == K0.ops_dekodierer_tour(n, d)
    assert K0.ops_dekodierer_tour(4, 2) == 92 == 2 * 2 * 2 + 3 * (2 * 2 + 4 * 2 + 4 * 4)


def test_h5_k_und_o_je_konstruktion():
    assert {n: KO.k_konstruktion(n) for n in C.GROESSEN} == {10: 45, 20: 190, 50: 1225, 100: 4950, 200: 19900}
    k = KO.kosten_je_konstruktion(20)
    assert k["O_nn"] == 1520 and KO.ops_nn(10) == 360
    assert k["O_zeiger"] == 246384 == 211200 + 35184 and KO.ops_zeiger(10) == 113704
    assert k["O_merkmal"] == 47242 and k["O_merkmal_mit_s"] == 47642       # ohne s (Plan) und mit s (Stück 3: zusätzlich n² für die Skala)
    assert round(k["f_zeiger"]) == 162 and round(k["f_merkmal"]) == 31
    assert KO.ops_merkmal(200) > KO.ops_zeiger(200) and KO.ops_zeiger(20) > KO.ops_merkmal(20)
    faktoren = [KO.kosten_je_konstruktion(n)[f] for n in C.GROESSEN for f in ("f_zeiger", "f_merkmal")]
    assert 25 < min(faktoren) and max(faktoren) < 320


def test_h5_geschlossene_formeln_des_formelblocks():
    for n in (10, 20, 50, 100, 200):
        d = 32
        o_zeiger = 2 * n * d + 8 * n * d * d + 2 * n * n * d + n * d * d + 2 * d * d + (n - 1) * (d * d + n * d + 4 * n)
        assert o_zeiger == KO.ops_zeiger(n)
        o_m = (n - 1) * (2 * n * n + n * (8 + math.log2(n))) + n * (n - 1) // 2 * 144
        assert int(o_m) == KO.ops_merkmal(n)


# ---------------------------------------------------------------- H6 Urteilsregel
B2 = [
    ("n=50 cluster A", +1.89, +0.30, +3.46, 5, "bricht"), ("n=50 cluster A+nt", +1.00, -0.44, +2.46, 4, "haelt"), ("n=50 cluster An", +0.99, -0.40, +2.48, 5, "haelt"),
    ("n=100 gleich A", +4.45, +2.07, +6.62, 5, "bricht"), ("n=100 gleich A+nt", +4.39, +1.99, +6.49, 5, "bricht"), ("n=100 gitter A", -1.38, -3.12, +0.38, 2, "haelt"),
    ("n=100 gleich B mit s", -5.51, -8.18, -2.98, 0, "schlaegt"), ("n=200 cluster B ohne s", -1.11, -2.88, +0.63, 0, "haelt"), ("n=200 cluster B mit s", -2.97, -5.00, -0.67, 0, "schlaegt"),
    ("n=200 gitter A", +7.70, +5.84, +9.71, 5, "bricht"),
]


@pytest.mark.parametrize("zelle,dif,lo,hi,seeds,soll", B2)
def test_h6_klasse_zelle_an_berichtswerten_der_vormessung(zelle, dif, lo, hi, seeds, soll):
    assert V.klasse_zelle(dif, lo, hi, seeds) == soll


def test_h6_zaehlungen_14_von_18_und_6_von_15():
    b3 = [(+2.08, 5), (+101.43, 5), (+2.02, 5), (-5.01, 0), (+110.95, 5), (-3.01, 0), (-3.30, 0), (+115.91, 5), (-1.54, 0), (+8.16, 5), (+245.59, 5), (+8.82, 5),
          (+2.24, 4), (+240.34, 5), (+3.87, 5), (+5.86, 5), (+265.36, 5), (+6.20, 5)]
    assert len(b3) == 18 and sum(1 for d, s in b3 if d > 0 and s >= V.seeds_noetig(5)) == 14
    b1 = [(-6.56, 0), (-3.08, 0), (-8.75, 0), (-7.96, 0), (-2.07, 0), (-9.13, 0), (-2.38, 0), (+1.89, 5), (-5.61, 0), (+4.45, 5), (+6.53, 5), (-1.38, 2), (+11.28, 5), (+19.06, 5), (+7.70, 5)]
    assert len(b1) == 15 and sum(1 for d, s in b1 if d > 0 and s >= 4) == 6


def test_seeds_noetig_ist_80_prozent_aufgerundet():
    assert [V.seeds_noetig(s) for s in (1, 2, 3, 4, 5)] == [1, 2, 3, 4, 4]

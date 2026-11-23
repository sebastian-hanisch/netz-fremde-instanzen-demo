"""Orakel-Tests (ORACLE-PLAYBOOK §2, Netze und Metaheuristiken): unabhängige Rechenwege für die Kernroutinen dieser Demo, alle unter 10 Sekunden.

(a) Gradienten-Check zuerst: Zeigernetz (d = 8, n = 6, eine Schicht, normierte Eingabe) und Merkmalsnetz (Politikverlust bei fester Skala) gegen zentrale endliche Differenzen.
(b) Held-Karp gegen Aufzählung (n = 5 bis 8) und gegen CP-SAT (n = 12, nur wenn ortools installiert ist).
(c) Merkmale x1 bis x7: Vektorfassung gegen die Schleifenfassung in `nfi_hand.py` an 20 Zufallsinstanzen und drei Schritten.
(d) Zeigernetz: Vorwärtspass (Log-Wahrscheinlichkeit einer Tour) gegen eine Fassung in reinem Python mit `math` bei n = 5, d = 4.
(e) Zählung: Merkmalszeilen je Tour gleich n(n-1)/2, Kandidatenvergleiche des Nächsten Nachbarn gleich n(n-1)/2.
(f) Bootstrap-Intervall gegen das t-Intervall bei normalverteilten Werten (Band).
(g) Testmenge und Trainingsmenge sind getrennt (Fehlerklasse „Testmenge überlappt Trainingsmenge“).
(h) Vektorziehung gleich 50 Einzelziehungen des skalaren SplitMix64."""

import itertools
import math

import numpy as np
import pytest

import nfi_constants as C
import nfi_exact as EXA
import nfi_gegner as GG
import nfi_hand as HA
import nfi_instances as I
import nfi_net as N
import nfi_scenario as SC
import nfi_stats as ST
import nfi_transform as TR
import nfik_featnet as KF
import nfik_mlp as MLP
import nfik_policy as KP
from nfi_rng import SplitMix64
from nfi_rngvec import UStream


# ---------------------------------------------------------------- (a) Gradienten-Check
def _zufalls_netz(d=8, n=6, M=2, K=2, seed=0):
    P = N.init_params(d, UStream(seed), layers=1, attn=True, ffn=True)
    X = I.normalize_unit(UStream(seed + 100).uniform((M, n, 2)))
    starts = np.zeros((M, K), dtype=np.int64)
    out = N.decode(P, N.encode(P, X)[0], starts, mode="sample", u_stream=UStream(seed + 200), keep=False)
    return P, X, starts, out["tours"]


def test_a_gradient_des_zeigernetzes_gegen_zentrale_differenzen():
    P, X, starts, tours = _zufalls_netz()
    w = np.array([[-0.5, 0.25], [0.75, -0.5]])

    def F(Pq):
        H, _ = N.encode(Pq, X, keep=False)
        return float((w * N.decode(Pq, H, starts, mode="forced", forced=tours, keep=False)["logp"]).sum())

    _, G = N.loss_and_grad(P, X, starts, w, tours)
    rng = np.random.default_rng(0)       # nur Auswahl der geprüften Einträge, nicht Teil der App
    groesste = max(float(np.abs(g).max()) for g in G.values())
    worst = 0.0
    for k in P:
        flach = P[k].reshape(-1)
        for idx in rng.choice(flach.size, size=min(6, flach.size), replace=False):
            alt = flach[idx]
            h = 1e-6
            flach[idx] = alt + h
            fp = F(P)
            flach[idx] = alt - h
            fm = F(P)
            flach[idx] = alt
            worst = max(worst, abs((fp - fm) / (2 * h) - G[k].reshape(-1)[idx]) / groesste)
    assert worst < 1e-6, worst


def test_a_gradient_des_merkmalsnetzes_bei_fester_skala_gegen_zentrale_differenzen():
    X = TR.testmenge("gleich", 8, 3)
    D = TR.abstaende(X)
    starts = np.zeros(3, dtype=np.int64)
    params = MLP.init_mlp(7, 16, UStream(1))
    _, _, _, rec = KF.rollout(D, np.full(3, C.S0), starts, params, us=UStream(9), mode="sample", record=True)
    coef = np.array([0.3, -0.2, 0.1])
    _, g = KP.policy_loss_grad(params, rec, coef)
    groesste = max(float(np.abs(x).max()) for x in g)
    worst = 0.0
    for k, p in enumerate(params):
        flach = p.reshape(-1)
        for idx in range(flach.size):
            alt = flach[idx]
            h = 1e-6
            flach[idx] = alt + h
            fp = KP.policy_loss_grad(params, rec, coef)[0]
            flach[idx] = alt - h
            fm = KP.policy_loss_grad(params, rec, coef)[0]
            flach[idx] = alt
            worst = max(worst, abs((fp - fm) / (2 * h) - g[k].reshape(-1)[idx]) / groesste)
    assert worst < 1e-6, worst


# ---------------------------------------------------------------- (b) Held-Karp
@pytest.mark.parametrize("n", [5, 6, 7, 8])
def test_b_held_karp_gegen_aufzaehlung(n):
    X = TR.testmenge("gleich", n, 20)
    D = TR.abstaende(X)
    for i in range(20):
        best = min(HA.tour_laenge(D[i], [0] + list(p)) for p in itertools.permutations(range(1, n)))
        assert EXA.held_karp(D[i]) == pytest.approx(best, abs=1e-12)
        L, t = EXA.held_karp_tour(D[i])
        assert sorted(t) == list(range(n)) and HA.tour_laenge(D[i], t) == pytest.approx(L)


def test_b_held_karp_gegen_cpsat_bei_n_12():
    cp = pytest.importorskip("ortools.sat.python.cp_model")
    X = TR.testmenge("cluster", 12, 2)
    D = TR.abstaende(X)
    for i in range(2):
        n = 12
        skala = 10 ** 6
        Di = np.rint(D[i] * skala).astype(int)
        m = cp.CpModel()
        arcs, lit = [], {}
        for a in range(n):
            for b in range(n):
                if a != b:
                    lit[a, b] = m.NewBoolVar(f"x{a}_{b}")
                    arcs.append((a, b, lit[a, b]))
        m.AddCircuit(arcs)
        m.Minimize(sum(int(Di[a, b]) * v for (a, b), v in lit.items()))
        s = cp.CpSolver()
        s.parameters.num_workers = 1
        s.parameters.max_time_in_seconds = 20
        assert s.Solve(m) == cp.OPTIMAL
        assert s.ObjectiveValue() / skala == pytest.approx(EXA.held_karp(D[i]), abs=2e-5)


# ---------------------------------------------------------------- (c) Merkmale
def test_c_merkmale_vektorfassung_gegen_schleife_an_zufallsinstanzen():
    n = 10
    X = TR.testmenge("cluster", n, 20)
    D = TR.abstaende(X)
    rng = np.random.default_rng(3)       # nur zufällige Touren zum Prüfen, nicht Teil der App
    for i in range(20):
        tour = np.concatenate([[0], rng.permutation(np.arange(1, n))])
        for mit_s in (True, False):
            s = HA.skala_s(D[i]) if mit_s else C.S0
            _, _, _, rec = KF.rollout(D[i][None], np.array([s]), np.array([0]), MLP.init_mlp(7, 16, UStream(2)), teacher=tour[None], record=True)
            for k in (0, 3, 6):
                Xk, a, unv = rec[k]
                cur = int(tour[k])
                frei = [j for j in range(n) if unv[0, j]]
                F = HA.merkmale(D[i], s, cur, 0, frei, k + 1)
                for j in frei:
                    assert np.allclose(Xk[0, j], F[j], atol=1e-12), (i, k, j)


# ---------------------------------------------------------------- (d) Vorwärtspass des Zeigernetzes in reinem Python
def _mm(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def py_logp_zeigernetz(P, x, tour):
    """log p der Tour (Start = tour[0]) für das Zeigernetz mit einer Schicht (Attention und Vorwärtsnetz) in reinem Python: Listen und math."""
    Pl = {k: v.tolist() for k, v in P.items()}
    n, d = len(x), len(Pl["W0"][0])
    H = [[sum(x[j][a] * Pl["W0"][a][k] for a in range(2)) + Pl["b0"][k] for k in range(d)] for j in range(n)]
    Q, Kk, V = _mm(H, Pl["Wq0"]), _mm(H, Pl["Wk0"]), _mm(H, Pl["Wv0"])
    A = []
    for i in range(n):
        s = [sum(Q[i][k] * Kk[j][k] for k in range(d)) / math.sqrt(d) for j in range(n)]
        m = max(s)
        e = [math.exp(v - m) for v in s]
        z = sum(e)
        A.append([v / z for v in e])
    AV = _mm(A, V)
    AO = _mm(AV, Pl["Wo0"])
    H1 = [[H[j][k] + AO[j][k] for k in range(d)] for j in range(n)]
    Z = [[sum(H1[j][a] * Pl["W10"][a][k] for a in range(d)) + Pl["b10"][k] for k in range(2 * d)] for j in range(n)]
    R = [[max(v, 0.0) for v in row] for row in Z]
    F2 = _mm(R, Pl["W20"])
    H2 = [[H1[j][k] + F2[j][k] + Pl["b20"][k] for k in range(d)] for j in range(n)]
    Wg, Wf, Wl = Pl["Wc"][0:d], Pl["Wc"][d:2 * d], Pl["Wc"][2 * d:3 * d]

    def mal(v, W):
        return [sum(v[i] * W[i][k] for i in range(d)) for k in range(d)]

    g = [sum(H2[j][k] for j in range(n)) / n for k in range(d)]
    cg = [a + b for a, b in zip(mal(g, Wg), mal(H2[tour[0]], Wf))]
    kp = [mal(H2[j], Pl["Wkd"]) for j in range(n)]
    logp, visited, cur = 0.0, {tour[0]}, tour[0]
    for a in tour[1:]:
        c = [p + q for p, q in zip(cg, mal(H2[cur], Wl))]
        u = {j: C.C_CLIP * math.tanh(sum(c[k] * kp[j][k] for k in range(d)) / math.sqrt(d)) for j in range(n) if j not in visited}
        logp += u[a] - math.log(sum(math.exp(v) for v in u.values()))
        visited.add(a)
        cur = a
    return logp


def test_d_vorwaertspass_des_zeigernetzes_gegen_reines_python():
    n, d = 5, 4
    for seed in range(3):
        P = N.init_params(d, UStream(seed), layers=1)
        X = UStream(seed + 50).uniform((2, n, 2))
        tours = np.array([[[0, 3, 1, 4, 2]], [[0, 1, 2, 3, 4]]])
        H, _ = N.encode(P, X, keep=False)
        out = N.decode(P, H, np.zeros((2, 1), dtype=np.int64), mode="forced", forced=tours, keep=False)
        for m in range(2):
            assert float(out["logp"][m, 0]) == pytest.approx(py_logp_zeigernetz(P, X[m].tolist(), tours[m, 0].tolist()), abs=1e-10)


# ---------------------------------------------------------------- (e) Zählung
@pytest.mark.parametrize("n", [5, 10, 20])
def test_e_zeilen_und_vergleiche_je_tour_sind_n_mal_n_minus_1_halbe(n):
    X = TR.testmenge("gleich", n, 3)
    D = TR.abstaende(X)
    _, _, zeilen, _ = KF.rollout(D, TR.skala_s(D), np.zeros(3, dtype=np.int64), MLP.init_mlp(7, 16, UStream(0)), mode="greedy")
    assert zeilen == 3 * n * (n - 1) // 2
    assert GG.nn_tour_k(D[0].tolist(), 0)[1] == n * (n - 1) // 2


# ---------------------------------------------------------------- (f) Bootstrap gegen t-Intervall
def test_f_bootstrap_intervall_gegen_t_intervall_als_band():
    x = UStream(11).normal((200,)) * 2.0 + 1.0
    lo, hi = ST.bootstrap_ci(x, 6)
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(len(x))
    halb = 1.972 * se
    assert abs(lo - (m - halb)) < 0.25 * halb and abs(hi - (m + halb)) < 0.25 * halb


# ---------------------------------------------------------------- (g) Testmenge getrennt von der Trainingsmenge
def test_g_testinstanzen_und_trainingsinstanzen_haben_keine_gemeinsamen_orte():
    test = TR.testmenge("gleich", 20, C.SEED_MAX + 1)
    train = I.train_batch(UStream(SC.derive(C.ZWECK_TRAIN_X, 20, 0)), 1000, 20)
    a = {tuple(np.round(p, 9)) for inst in test for p in inst}
    b = {tuple(np.round(p, 9)) for inst in train for p in inst}
    assert len(a) == 100 * 20 and len(b) == 1000 * 20 and not (a & b)


# ---------------------------------------------------------------- (h) Strom
def test_h_vektorziehung_gleicht_50_einzelziehungen():
    for seed in (0, 7, 12345):
        v = UStream(seed).uniform((50,))
        r = SplitMix64(seed)
        assert np.array_equal(v, np.array([r.random() for _ in range(50)]))


# ---------------------------------------------------------------- (b2) Referenzwerkzeug
def test_b_gls_referenz_gleicht_held_karp_bei_n_12():
    """Das Werkzeug der Referenzen (OR-Tools Guided Local Search, `tools/nfi_make_refs.py`) findet bei n = 12 das Optimum von Held-Karp."""
    pytest.importorskip("ortools")
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("nfi_make_refs", Path(__file__).resolve().parent.parent / "tools" / "nfi_make_refs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    X = TR.testmenge("gleich", 12, 2)
    D = TR.abstaende(X)
    for i in range(2):
        L, tour = m.gls_laenge(X[i], 1.0)
        assert sorted(tour) == list(range(12)) and L == pytest.approx(EXA.held_karp(D[i]), abs=1e-5)

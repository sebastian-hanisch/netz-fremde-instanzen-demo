"""Trainingsschleifen (`nfi_trainloops.py`): ohne Veränderung der Eingabe schrittgleich zu den Quellen, jede Variante weicht von der Nachbarvariante ab (Zweig-Test), Netze aus der Datei.
Kein Test trainiert mehr als 20 Schritte."""

import numpy as np
import pytest

import nfi_constants as C
import nfi_models as MO
import nfi_net as N
import nfi_train as T
import nfi_trainloops as TL
import nfik_policy as KP


def _diff(P, Q):
    return max(float(np.abs(P[k] - Q[k]).max()) for k in P)


def test_zeiger_roh_ist_schrittgleich_zu_nfi_train():
    P1, _, _ = T.train(20, 8, init_seed=2, data_seed=2, eval_every=0)
    P2, cnt = TL.train_zeiger("roh", 2, steps=8)
    assert _diff(P1, P2) == 0.0 and cnt["touren"] == 8 * C.BATCH


def test_merkmal_mit_s_ist_schrittgleich_zu_train_featnet():
    p1, _, _ = KP.train_featnet(6, seed=1)
    p2, cnt = TL.train_merkmal("mit", 1, steps=6)
    assert all(np.array_equal(a, b) for a, b in zip(p1, p2)) and cnt["touren"] == 6 * 256


def test_jede_variante_weicht_ab_und_die_wirkung_ist_schaltbar():
    roh, _ = TL.train_zeiger("roh", 2, steps=8)
    for eingabe in ("normiert", "augmentiert"):
        P, _ = TL.train_zeiger(eingabe, 2, steps=8)
        assert _diff(roh, P) > 1e-4, eingabe
    mix, cnt = TL.train_zeiger("normiert", 2, steps=8, groessen=(10, 20, 50))
    assert _diff(roh, mix) > 1e-4 and cnt["schritte"] == 8
    mit, _ = TL.train_merkmal("mit", 1, steps=6)
    ohne, _ = TL.train_merkmal("ohne", 1, steps=6)
    assert max(float(np.abs(a - b).max()) for a, b in zip(mit, ohne)) > 1e-3


def test_eingabe_veraendern_normiert_und_augmentiert():
    from nfi_rngvec import UStream
    X = UStream(0).uniform((4, 20, 2))
    Xn = TL.eingabe_veraendern(X, "normiert")
    assert Xn.min() == pytest.approx(0.0) and Xn.max() == pytest.approx(1.0)
    Xa = TL.eingabe_veraendern(X, "augmentiert", UStream(5))
    assert Xa.shape == X.shape and not np.allclose(Xa, X)
    with pytest.raises(ValueError):
        TL.eingabe_veraendern(X, "xyz")


def test_modelle_aus_den_dateien_sind_gueltig_und_gepaart():
    for netz in ("A", "Bmit"):
        for s in range(C.NETZ_SEEDS):
            P, meta = MO.lade(netz, s)
            assert (N.n_params(P) == C.ZEIGER_PARAMS) if netz == "A" else (sum(p.size for p in P) == C.MERKMAL_PARAMS)
            assert int(meta["steps"]) == (C.SCHRITTE_ZEIGER if netz == "A" else C.SCHRITTE_MERKMAL)
    assert MO.pfad("Ant", 0) == MO.pfad("A", 0)        # A+nt benutzt die Gewichte von A

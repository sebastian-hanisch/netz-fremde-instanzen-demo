"""Instanzen und Wechsel (`nfi_transform.py`): Wechsel, Testmengen aller Familien, Nächster Nachbar und Tourlängen für viele Instanzen, Invarianzen."""

import numpy as np
import pytest

import nfi_constants as C
import nfi_gegner as GG
import nfi_instances as I
import nfi_transform as TR


@pytest.mark.parametrize("familie", C.FAMILIEN)
@pytest.mark.parametrize("n", [10, 20, 50, 100, 200])
def test_testmenge_hat_die_richtige_form_und_ist_endlich(familie, n):
    X = TR.testmenge(familie, n, 3)
    assert X.shape == (3, n, 2) and np.isfinite(X).all()
    assert np.array_equal(TR.testmenge(familie, n, 2, ab=1)[0], X[1])        # Zufalls-Seed 1 ist Zufalls-Seed 1


def test_wechsel_arten_und_fehler():
    X = TR.testmenge("gleich", 10, 2)
    assert np.array_equal(TR.wechsel(X, "basis"), X)
    assert np.allclose(TR.wechsel(X, "klein"), 0.25 * X) and np.allclose(TR.wechsel(X, "gross"), 10 * X)
    assert np.allclose(TR.wechsel(X, "versch"), X + np.array([3.0, 2.0]))
    assert np.allclose(TR.wechsel_frei(X, 2.0, -1, 4), 2 * X + np.array([-1.0, 4.0]))
    with pytest.raises(ValueError):
        TR.wechsel(X, "gedreht")


def test_nn_touren_gleicht_der_kopie_der_quelle_und_ist_eine_permutation():
    X = TR.testmenge("cluster", 20, 20)
    D = TR.abstaende(X)
    T = TR.nn_touren(D)
    assert TR.ist_permutation(T)
    for i in range(20):
        assert T[i].tolist() == GG.nn_tour_k(D[i].tolist(), 0)[0]
    L = TR.tour_laengen(D, T)
    assert np.allclose(L, [sum(D[i][T[i][k], T[i][(k + 1) % 20]] for k in range(20)) for i in range(20)])


@pytest.mark.parametrize("familie", C.FAMILIEN)
def test_nn_und_normierung_sind_invariant_gegen_skalieren_und_verschieben(familie):
    X = TR.testmenge(familie, 20, 20)
    T0 = TR.nn_touren(TR.abstaende(X))
    for art in ("klein", "gross", "versch"):
        Xw = TR.wechsel(X, art)
        assert np.array_equal(TR.nn_touren(TR.abstaende(Xw)), T0), art
        assert np.allclose(I.normalize_unit(Xw), I.normalize_unit(X), atol=1e-12), art
        fak = {"klein": 0.25, "gross": 10.0, "versch": 1.0}[art]
        assert np.allclose(TR.tour_laengen(TR.abstaende(Xw), T0), fak * TR.tour_laengen(TR.abstaende(X), T0))


def test_skala_s_und_ihre_invarianz_der_verhaeltnisse():
    X = TR.testmenge("gleich", 20, 5)
    D = TR.abstaende(X)
    s = TR.skala_s(D)
    assert s.shape == (5,) and np.all(s > 0)
    Dg = TR.abstaende(10 * X)
    assert np.allclose(Dg / TR.skala_s(Dg)[:, None, None], D / s[:, None, None], atol=1e-12)
    assert TR.skala_s(D[0]) == pytest.approx(s[0])


def test_ist_permutation_erkennt_fehler():
    assert TR.ist_permutation(np.array([[0, 2, 1], [1, 0, 2]]))
    assert not TR.ist_permutation(np.array([[0, 2, 2]]))


def test_normierung_wacht_gegen_division_durch_null():
    X = np.zeros((1, 5, 2)) + 0.3
    assert np.array_equal(I.normalize_unit(X), np.zeros((1, 5, 2)))

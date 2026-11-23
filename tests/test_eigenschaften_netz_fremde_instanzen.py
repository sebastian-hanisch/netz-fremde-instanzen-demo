"""Eigenschaftstests (ORACLE §5.2) an den vorgerechneten Netzen: Skala und Lage ändern die Touren der invarianten Netze nicht (Konstruktion), das Zeigernetz mit roher Eingabe bricht bei
Faktor 10 (Gegenprobe: sonst wäre der Schalter tot), B ohne s ist gegen Verschieben invariant und gegen Faktor 10 nicht; jede Tour ist eine Permutation."""

import numpy as np
import pytest

import nfi_constants as C
import nfi_eval as EV
import nfi_models as MO
import nfi_transform as TR

FAM_N = [("gleich", 20), ("cluster", 20), ("gitter", 20), ("gleich", 50)]


def _basis(fam, n, m=12):
    X = TR.testmenge(fam, n, m)
    return X, TR.abstaende(X)


@pytest.mark.parametrize("netz", ["Ant", "Bmit"])
@pytest.mark.parametrize("fam,n", FAM_N)
def test_invariante_netze_liefern_unter_jedem_wechsel_dieselbe_tour(netz, fam, n):
    if not MO.vorhanden("A" if netz == "Ant" else netz):
        pytest.skip("Modelle fehlen")
    X, D = _basis(fam, n)
    modelle = [MO.lade(netz, s)[0] for s in (0, 1)]
    T0 = EV.laengen_netz(netz, modelle, X, D)[1]
    for art in ("klein", "gross", "versch"):
        T = EV.laengen_netz(netz, modelle, TR.wechsel(X, art), D)[1]
        assert (T == T0).all(-1).mean() >= 0.99, (netz, art)       # Anteil gleicher Touren (Konstruktion: 1,00; Rundung in der letzten Stelle ließe einen Gleichstand kippen)


def test_an_und_amix_sind_normiert_invariant():
    for netz in ("An", "Amix"):
        if not MO.vorhanden(netz):
            pytest.skip(f"{netz}: Modelle fehlen")
        X, D = _basis("gleich", 20)
        modelle = [MO.lade(netz, 0)[0]]
        T0 = EV.laengen_netz(netz, modelle, X, D)[1]
        T1 = EV.laengen_netz(netz, modelle, TR.wechsel(X, "gross"), D)[1]
        assert (T0 == T1).all()


def test_zeigernetz_roh_aendert_die_tour_bei_faktor_10_zweig_test():
    X, D = _basis("gleich", 20)
    modelle = [MO.lade("A", s)[0] for s in (0, 1, 2)]
    T0 = EV.laengen_netz("A", modelle, X, D)[1]
    T1 = EV.laengen_netz("A", modelle, TR.wechsel(X, "gross"), D)[1]
    assert (T0 == T1).all(-1).mean() < 1.0
    L0 = TR.tour_laengen(D, T0[0]).mean()
    L1 = TR.tour_laengen(D, T1[0]).mean()
    assert L1 > L0 + 0.5            # bei Faktor 10 deutlich längere Touren (Zufallsniveau liegt bei einem Mehrfachen)


def test_b_ohne_s_ist_gegen_verschieben_invariant_und_gegen_faktor_10_nicht():
    if not MO.vorhanden("Bohne"):
        pytest.skip("Bohne: Modelle fehlen")
    X, D = _basis("gleich", 20)
    modelle = [MO.lade("Bohne", s)[0] for s in (0, 1, 2)]
    T0 = EV.laengen_netz("Bohne", modelle, X, D)[1]
    Tv = EV.laengen_netz("Bohne", modelle, TR.wechsel(X, "versch"), D)[1]
    Tg = EV.laengen_netz("Bohne", modelle, TR.wechsel(X, "gross"), D)[1]
    assert (T0 == Tv).all(-1).mean() >= 0.99 and (T0 == Tg).all(-1).mean() < 1.0


def test_laenge_unter_wechsel_ist_faktor_mal_laenge_fuer_den_naechsten_nachbarn():
    X, D = _basis("gleich", 20)
    T = TR.nn_touren(D)
    Dg = TR.abstaende(TR.wechsel(X, "gross"))
    assert np.allclose(TR.tour_laengen(Dg, T), C.FAKTOR_GROSS * TR.tour_laengen(D, T))


def test_umnummerieren_aendert_den_naechsten_nachbarn_nur_im_gleichstand():
    X, D = _basis("gleich", 20, 8)
    rng = np.random.default_rng(1)
    for i in range(8):
        perm = np.concatenate([[0], 1 + rng.permutation(19)])
        Xp = X[i][perm][None]
        L1 = TR.tour_laengen(D[i][None], TR.nn_touren(D[i][None]))[0]
        L2 = TR.tour_laengen(TR.abstaende(Xp), TR.nn_touren(TR.abstaende(Xp)))[0]
        assert L1 == pytest.approx(L2)

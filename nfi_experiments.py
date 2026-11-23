"""Experimente: Lesen der eingefrorenen Messung (`nfi_frozen.json`), Zählungen über die Zellen, Live-Rechnungen (Instanz, Wechsel, Größe) und Teilstichproben (E4).

`Frozen` hält die Messung und rechnet Zellenstatistiken (`nfi_eval.zelle`) einmal je Zelle. Wechsel: die gegen Skala und Lage invarianten Netze (A+nt, An, B mit s, Amix) werden unter den
Wechseln nicht noch einmal gerechnet, ihre Zelle ist die Basiszelle (Gleichheit folgt aus der Konstruktion; der Anteil gleicher Touren steht unter `invarianz`)."""

import json
from pathlib import Path

import numpy as np

import nfi_constants as C
import nfi_eval as EV
import nfi_evaluation as VD
import nfi_exact as EXA
import nfi_kosten as KO
import nfi_models as MO
import nfi_rngvec as RV
import nfi_scenario as SC
import nfi_transform as TR

FROZEN_PFAD = Path(__file__).resolve().parent / "nfi_frozen.json"
NETZ_IM_WECHSEL = {"A": "A", "Aaug": "Aaug", "Bohne": "Bohne"}      # Netze, die unter einem Wechsel eigene Längen haben


class Frozen:
    """Die eingefrorene Messung mit zwischengespeicherten Zellenstatistiken."""

    def __init__(self, daten: dict):
        self.d = daten
        self._z = {}

    @classmethod
    def laden(cls, pfad: Path = FROZEN_PFAD):
        """`Frozen` aus der Datei oder None, wenn sie fehlt."""
        return cls(json.loads(Path(pfad).read_text(encoding="utf-8"))) if Path(pfad).exists() else None

    # ------------------------------------------------------------------------------------------------ Zugriff
    def netze(self) -> list:
        return list(self.d["meta"]["netze"])

    def hat_netz(self, netz: str) -> bool:
        return netz in self.d["meta"]["netze"]

    def m(self, n: int) -> int:
        return int(self.d["instanzen"][str(n)])

    def basis(self, fam: str, n: int) -> dict:
        return self.d["base"][f"{fam}_{n}"]

    def ref(self, fam: str, n: int) -> np.ndarray:
        return np.array(self.basis(fam, n)["ref"])

    def nn(self, fam: str, n: int) -> np.ndarray:
        return np.array(self.basis(fam, n)["nn"])

    def laengen(self, fam: str, n: int, netz: str, art: str = "basis"):
        """Längen (S, M) des Netzes in der Zelle (Familie, n, Wechsel) auf der Basisinstanz; None, wenn nicht gemessen."""
        if art == "basis" or netz not in NETZ_IM_WECHSEL:
            L = self.basis(fam, n)["modelle"].get(netz)
        else:
            L = self.d["wechsel"].get(f"{fam}_{n}_{art}", {}).get("modelle", {}).get(netz)
        return None if L is None else np.array(L)

    def zelle(self, fam: str, n: int, netz: str, art: str = "basis"):
        """Statistik der Zelle (`nfi_eval.zelle`) oder None."""
        k = (fam, n, netz, art)
        if k not in self._z:
            L = self.laengen(fam, n, netz, art)
            self._z[k] = None if L is None else EV.zelle(L, self.nn(fam, n), self.ref(fam, n))
        return self._z[k]

    def delta_je_seed(self, fam: str, n: int, netz: str, art: str = "basis"):
        """Einzelwerte Delta (S, M) in Prozentpunkten der Referenz."""
        L = self.laengen(fam, n, netz, art)
        return None if L is None else 100.0 * (L - self.nn(fam, n)[None, :]) / self.ref(fam, n)[None, :]

    def seeds_unter_nn(self, netz: str, fam: str = "gleich", n: int = 20) -> int:
        """Zahl der Zufalls-Seeds, deren mittlerer Unterschied zum Nächsten Nachbarn auf der Trainingszelle (n = 20 gleichverteilt) unter null liegt (Modellgüte-Marke)."""
        d = self.delta_je_seed(fam, n, netz)
        return 0 if d is None else int((d.mean(1) < 0).sum())

    def luecke_netz(self, fam: str, n: int, netz: str) -> float | None:
        z = self.zelle(fam, n, netz)
        return None if z is None else z["luecke"]

    def luecke_nn(self, fam: str, n: int) -> float:
        return EV.luecke_nn(self.nn(fam, n), self.ref(fam, n))

    def luecke_nn2opt(self, fam: str, n: int) -> float | None:
        w = self.basis(fam, n).get("nn2opt")
        if not w:
            return None
        r = self.ref(fam, n)[:len(w)]
        return float((100.0 * (np.array(w) - r) / r).mean())

    # ------------------------------------------------------------------------------------------------ Zählungen
    def wechsel_klassen(self, netz: str, groessen=C.WECHSEL_GROESSEN) -> dict:
        """{(Familie, n, Wechsel): Urteil} der 18 Wechselzellen; die invarianten Netze (A+nt, An, B mit s, Amix) zählen mit ihrer Basiszelle (Gleichheit per Konstruktion)."""
        out = {}
        for n in groessen:
            for fam in C.FAMILIEN:
                for art in ("klein", "gross", "versch"):
                    z = self.zelle(fam, n, netz, art)
                    out[(fam, n, art)] = None if z is None else z["klasse"]
        return out

    def basis_klassen(self, netz: str, n_min: int = 0) -> dict:
        """{(Familie, n): Urteil} der Basiszellen (15) mit n >= n_min."""
        out = {}
        for n in C.GROESSEN:
            if n < n_min:
                continue
            for fam in C.FAMILIEN:
                z = self.zelle(fam, n, netz)
                out[(fam, n)] = None if z is None else z["klasse"]
        return out

    def wechsel_luecke_gleich_basis(self, netz: str, art_liste=("klein", "gross", "versch")) -> int:
        """Zahl der Wechselzellen (n = 20, 50 x Familien x Wechsel), in denen die Lücke des Netzes um mindestens 2 Prozentpunkte über der der Basiszelle liegt (Regel b: bleibt nicht auf der Basislücke)."""
        k = 0
        for n in C.WECHSEL_GROESSEN:
            for fam in C.FAMILIEN:
                b = self.zelle(fam, n, netz)
                for art in art_liste:
                    z = self.zelle(fam, n, netz, art)
                    if b is not None and z is not None and z["luecke"] - b["luecke"] >= C.MARKE_HAELT_LUECKE:
                        k += 1
        return k


# ---------------------------------------------------------------------------------------------------------------- Live
def referenz_live(fam: str, n: int, seeds: list[int], frozen: Frozen | None) -> np.ndarray | None:
    """Referenzlängen für die Instanzen mit den Zufalls-Seeds `seeds`: aus der eingefrorenen Messung (bei n >= 50 beste bekannte Lösung, sonst exakt); für n <= 12 sonst Held-Karp live;
    sonst None."""
    if frozen is not None and max(seeds) < frozen.m(n):
        return frozen.ref(fam, n)[seeds]
    if n <= 12:
        X = np.stack([TR.zufalls_instanz(fam, n, s) for s in seeds])
        D = TR.abstaende(X)
        return np.array([EXA.held_karp(D[i]) for i in range(len(X))])
    return None


def live_zelle(fam: str, n: int, netz: str, seeds_instanz: list[int], zahl_seeds: int, art_oder_frei, frozen: Frozen | None):
    """Live-Zelle: Netz `netz` (Gewichts-Zufalls-Seeds 0 bis zahl_seeds − 1) auf den Instanzen `seeds_instanz` unter dem Wechsel (`art_oder_frei`: Name oder (faktor, vx, vy)).
    Gibt `nfi_eval.zelle` oder None ohne Referenz."""
    ref = referenz_live(fam, n, seeds_instanz, frozen)
    if ref is None:
        return None
    X = np.stack([TR.zufalls_instanz(fam, n, s) for s in seeds_instanz])
    D = TR.abstaende(X)
    Xw = TR.wechsel(X, art_oder_frei) if isinstance(art_oder_frei, str) else TR.wechsel_frei(X, *art_oder_frei)
    L, _ = EV.laengen_netz(netz, MO.lade_alle(netz, range(zahl_seeds)), Xw, D)
    nn = TR.tour_laengen(D, TR.nn_touren(D))
    return EV.zelle(L, nn, ref)


def live_o_zelle(n: int, netze: tuple, seeds: int, m: int) -> float:
    """Grobe Operationen einer Live-Rechnung (Obergrenze `C.LIVE_O_MAX`, `nfi_kosten.cap_live`)."""
    return KO.live_o(n, netze, seeds, m)


# ---------------------------------------------------------------------------------------------------------------- Teilstichproben (E4)
def teilstichproben(delta: np.ndarray, m: int, R: int, ziehungen: int = C.E4_BOOT, schluessel: int = 0) -> dict:
    """Zieht R Teilstichproben vom Umfang m (ohne Zurücklegen) aus den Einzelwerten Delta (S, M) einer Zelle und urteilt je Teilstichprobe nach `klasse_zelle` mit einem Bootstrap-Intervall
    (`ziehungen` Ziehungen über die m Instanzen). SplitMix64 (`nfi_rngvec`). Rückgabe {"v" (R,), "klassen" [R], "breite" (mittlere halbe Intervallbreite), "R", "m"}."""
    S, M = delta.shape
    m = max(2, min(int(m), M))
    R = int(R)
    us = RV.UStream(SC.derive(C.ZWECK_TEILPROBE, M, m, schluessel))
    idx = RV.choice_without_replacement(us, R, M, m)                               # (R, m)
    boot = np.minimum((RV.uniform_array(us, (R, ziehungen, m)) * m).astype(np.int64), m - 1)
    je_inst = delta.mean(0)[idx]                                                    # (R, m)
    v = je_inst.mean(1)
    mittel = np.take_along_axis(np.broadcast_to(je_inst[:, None, :], (R, ziehungen, m)), boot, axis=2).mean(2)
    lo, hi = np.percentile(mittel, 2.5, axis=1), np.percentile(mittel, 97.5, axis=1)
    seeds_pos = (delta[:, idx].mean(2) > 0).sum(0)                                  # (R,)
    klassen = [VD.klasse_zelle(float(v[r]), float(lo[r]), float(hi[r]), int(seeds_pos[r]), S) for r in range(R)]
    return {"v": v, "klassen": klassen, "breite": float(((hi - lo) / 2).mean()), "R": R, "m": m, "lo": lo, "hi": hi}


def e4_cap(m: int, R: int) -> int:
    """Begrenzt R so, dass R · 400 · m höchstens `C.E4_ZIEHUNGEN_MAX` Ziehungen braucht."""
    return max(1, min(int(R), C.E4_ZIEHUNGEN_MAX // (C.E4_BOOT * max(1, int(m)))))

"""Auswertung: Touren der Netze auf einer Menge von Instanzen, Längen auf der Basisinstanz, gepaarte Unterschiede, Zellenstatistik.

Einsatz der Netze (eine greedy-Tour je Instanz, Start Stadt 0, K = n(n-1)/2):
* Zeigernetz A und Aaug sehen die Koordinaten, wie sie sind; A+nt (`Ant`), An und Amix sehen `normalize_unit(X)` (Einheitsquadrat).
* Merkmalsnetz `Bmit` benutzt die Skala s der gesehenen Instanz, `Bohne` die feste Skala s0 (`nfi_constants.S0`).
Die Instanz, die das Netz sieht (`X_welt`), kann gegen die Basisinstanz skaliert und verschoben sein; **bewertet wird immer auf der Basisinstanz** (`D_basis`): die Tour ist eine
Permutation, ihre Länge hängt nur von der Reihenfolge ab, und der Nächste Nachbar wählt auf jeder gleichmäßig skalierten und verschobenen Fassung dieselbe Tour.

Zellenstatistik (`zelle`): Unterschied Delta = 100 (L_Netz - L_NN) / L_ref je Instanz und Seed, gepaart über Instanzen; das Mittel über die Seeds je Instanz geht in ein 95-%-Bootstrap-Intervall
über Instanzen (SplitMix64-Blockstrom, `nfi_stats.bootstrap_ci`); „Seeds über null“ zählt die Zufalls-Seeds, deren mittlerer Unterschied über null liegt."""

import numpy as np

import nfi_constants as C
import nfi_evaluation as EV
import nfi_instances as I
import nfi_net as N
import nfi_stats as ST
import nfi_transform as TR
import nfik_featnet as KF

CHUNK_ZEIGER = {10: 100, 20: 100, 50: 50, 100: 25, 200: 15}
CHUNK_MERKMAL = {10: 100, 20: 100, 50: 50, 100: 25, 200: 10}


def eingabe_netz(netz: str, X_welt: np.ndarray) -> np.ndarray:
    """Koordinaten (M, n, 2), die das Zeigernetz sieht: normiert für A+nt, An und Amix, sonst roh."""
    return I.normalize_unit(X_welt) if netz in ("Ant", "An", "Amix") else np.asarray(X_welt, dtype=np.float64)


def touren_zeiger(netz: str, P: dict, X_welt: np.ndarray) -> np.ndarray:
    """Greedy-Touren (M, n) eines Zeigernetzes auf der gesehenen Instanz `X_welt` (M, n, 2), Start Stadt 0."""
    X = eingabe_netz(netz, X_welt)
    n = X.shape[1]
    out = []
    ch = CHUNK_ZEIGER.get(n, 25)
    for i in range(0, len(X), ch):
        Xc = X[i:i + ch]
        out.append(N.greedy_tours(P, Xc, np.zeros((len(Xc), 1), dtype=np.int64), chunk=ch)[:, 0, :])
    return np.concatenate(out)


def touren_merkmal(netz: str, params, X_welt: np.ndarray) -> tuple[np.ndarray, int]:
    """Greedy-Touren (M, n) des Merkmalsnetzes auf der gesehenen Instanz und die Zahl der Merkmalszeilen; `Bmit` mit s der Instanz, `Bohne` mit der festen Skala s0."""
    X = np.asarray(X_welt, dtype=np.float64)
    n = X.shape[1]
    ch = CHUNK_MERKMAL.get(n, 10)
    out, zeilen = [], 0
    for i in range(0, len(X), ch):
        D = TR.abstaende(X[i:i + ch])
        s = TR.skala_s(D) if netz == "Bmit" else np.full(len(D), C.S0)
        T, _, z, _ = KF.rollout(D, s, np.zeros(len(D), dtype=np.int64), params, mode="greedy")
        out.append(T)
        zeilen += z
    return np.concatenate(out), zeilen


def laengen_netz(netz: str, modelle: list, X_welt: np.ndarray, D_basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Längen (S, M) und Touren (S, M, n) eines Netzes über seine Seeds (`modelle`: Parameter je Seed) auf den gesehenen Instanzen, gemessen auf `D_basis` (M, n, n).
    Jede Tour wird als Permutation geprüft."""
    Ls, Ts = [], []
    for params in modelle:
        T = touren_zeiger(netz, params, X_welt) if netz in C.ZEIGER_NETZE else touren_merkmal(netz, params, X_welt)[0]
        if not TR.ist_permutation(T):
            raise AssertionError(f"Tour von {netz} ist keine Permutation")
        Ts.append(T)
        Ls.append(TR.tour_laengen(D_basis, T))
    return np.stack(Ls), np.stack(Ts)


def zelle(L_netz: np.ndarray, L_nn: np.ndarray, ref: np.ndarray) -> dict:
    """Statistik einer Zelle. `L_netz` (S, M) Längen auf der Basisinstanz, `L_nn` (M,), `ref` (M,). Gibt Unterschied v zum Nächsten Nachbarn (Prozentpunkte), Intervall, Zahl der Seeds mit v > 0,
    Lücke zur Referenz (Mittel in Prozent), Urteil und Zahlen."""
    L_netz = np.asarray(L_netz, dtype=np.float64)
    S, M = L_netz.shape
    delta = 100.0 * (L_netz - L_nn[None, :]) / ref[None, :]
    je_inst = delta.mean(0)
    v = float(je_inst.mean())
    ci = ST.bootstrap_ci(je_inst, C.ZWECK_BOOT_ZELLE)
    lo, hi = (ci if ci is not None else (None, None))
    seeds_pos = int((delta.mean(1) > 0).sum())
    luecke = float((100.0 * (L_netz - ref[None, :]) / ref[None, :]).mean())
    klasse = EV.klasse_zelle(v, lo, hi, seeds_pos, S) if lo is not None else None
    return {"v": v, "lo": lo, "hi": hi, "seeds_pos": seeds_pos, "seeds": S, "m": M, "luecke": luecke, "klasse": klasse}


def luecke_nn(L_nn: np.ndarray, ref: np.ndarray) -> float:
    """Mittlere Lücke des Nächsten Nachbarn in Prozent."""
    return float((100.0 * (L_nn - ref) / ref).mean())

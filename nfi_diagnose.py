"""Diagnosen (Hinweise auf den Mechanismus, **keine Ursachenmessung**): Sättigung der Zeiger-Logits, Kappung der Entfernungsmerkmale, Skala der Instanz gegen die feste Skala.

* `saettigung_anteil`: Anteil der Zeiger-Logits u = C tanh(.) mit Betrag über 0,964 C (also tanh nahe 1) unter den noch freien Kandidaten, entlang der Greedy-Touren des Zeigernetzes.
  Sind fast alle Logits am Rand, unterscheiden sich die Kandidaten kaum noch (Gleichstand, `argmax` nimmt den kleinsten Index).
* `kappung_anteil`: Anteil der Kandidaten unter den noch freien, deren Merkmal x1, x3 oder x4 bei 6 gekappt ist, und derer mit x5 gleich 0, entlang der Greedy-Touren des Merkmalsnetzes.
* `s_verhaeltnis`: mittlere Skala s der Instanzen geteilt durch die feste Skala s0."""

import numpy as np

import nfi_constants as C
import nfi_net as N
import nfik_featnet as KF

GRENZE_TANH = 0.964


def saettigung_anteil(P: dict, X_ein: np.ndarray) -> float:
    """Anteil gesättigter Zeiger-Logits (|u| > 0,964 C) unter den freien Kandidaten, Greedy-Touren ab Stadt 0, Instanzen X_ein (M, n, 2) wie das Netz sie sieht."""
    X_ein = np.asarray(X_ein, dtype=np.float64)
    M, n, _ = X_ein.shape
    starts = np.zeros((M, 1), dtype=np.int64)
    H, _ = N.encode(P, X_ein, keep=False)
    out = N.decode(P, H, starts, mode="greedy", keep=True)
    us = out["cache"]["us"][:, :, 0, :]                # (n-1, M, n)
    pos = np.argsort(out["tours"][:, 0, :], axis=-1)     # Position jeder Stadt in der Tour (M, n)
    t = np.arange(n - 1)[:, None, None]
    frei = pos[None, :, :] > t                          # nach Schritt t sind die Städte mit Position <= t besucht
    gesaettigt = np.abs(us) > GRENZE_TANH * C.C_CLIP
    return float((gesaettigt & frei).sum() / frei.sum())


def kappung_anteil(params, D_welt: np.ndarray, s: np.ndarray) -> dict:
    """Anteile gekappter Merkmale des Merkmalsnetzes entlang seiner Greedy-Touren: {"x1", "x3", "x4", "x5_null"} unter den freien Kandidaten. `D_welt` (R, n, n), `s` (R,)."""
    R = D_welt.shape[0]
    T, _, _, rec = KF.rollout(D_welt, s, np.zeros(R, dtype=np.int64), params, mode="greedy", record=True)
    zaehler = np.zeros(4)
    gesamt = 0
    for X, _, unv in rec:
        gesamt += int(unv.sum())
        zaehler += np.array([
            ((X[..., 0] >= 6.0) & unv).sum(), ((X[..., 2] >= 6.0) & unv).sum(), ((X[..., 3] >= 6.0) & unv).sum(), ((X[..., 4] == 0.0) & unv).sum(),
        ])
    a = zaehler / gesamt
    return {"x1": float(a[0]), "x3": float(a[1]), "x4": float(a[2]), "x5_null": float(a[3])}


def s_verhaeltnis(D: np.ndarray) -> float:
    """Mittleres s der Instanzen (R, n, n) geteilt durch die feste Skala s0."""
    n = D.shape[-1]
    return float((D + np.diag(np.full(n, np.inf))).min(-1).mean() / C.S0)

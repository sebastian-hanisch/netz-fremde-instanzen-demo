"""Trainingsschleifen dieser Demo (nur von `tools/` und den Tests benutzt, nie von der App).

Die Kopie `nfi_train.train` hat keinen Schalter für die Eingabe des Zeigernetzes (Stück 1: der Aufrufer normiert), und `nfik_policy.train_featnet` keine Wahl der Skala s.
Darum stehen hier die Schleifen der Varianten, die diese Demo braucht, aus den kopierten Bausteinen (`encode`, `decode`, `decode_backward`, `encode_backward`, `Adam`,
`rollout`, `policy_loss_grad`):

* Zeigernetz A (rohe Eingabe), An (Eingabe auch im Training normiert), Aaug (im Training je Instanz skaliert und verschoben), Amix (normiert, Größe je Schritt aus {10, 20, 50});
* Merkmalsnetz mit s (Skala der Instanz) und ohne s (feste Skala s0).

Die Länge im Verlust ist immer die Länge auf der **Originalinstanz** (nicht auf der veränderten Eingabe), so dass die Belohnung nicht mit dem Faktor skaliert.
Ohne Veränderung der Eingabe (`eingabe="roh"`) ist die Schleife schrittgleich zu `nfi_train.train` (Test `test_trainloops.py`), mit Skala s der Instanz schrittgleich zu
`nfik_policy.train_featnet`."""

import time

import numpy as np

import nfi_constants as C
import nfi_instances as I
import nfi_net as N
import nfi_scenario as SC
import nfi_train as T
import nfik_constants as KC
import nfik_featnet as KF
import nfik_mlp as MLP
import nfik_policy as KP
from nfi_rngvec import UStream

EINGABEN = ("roh", "normiert", "augmentiert")


def eingabe_veraendern(X: np.ndarray, art: str, us_aug: UStream | None = None) -> np.ndarray:
    """Eingabe des Zeigernetzes im Training: roh, normiert (Einheitsquadrat) oder augmentiert (Faktor logarithmisch gleichverteilt in [0,1; 10], Verschiebung gleichverteilt in [-3; 3] je Achse,
    je Instanz; Ziehung (m, 3) aus `us_aug`)."""
    if art == "roh":
        return X
    if art == "normiert":
        return I.normalize_unit(X)
    if art == "augmentiert":
        m = X.shape[0]
        u = us_aug.uniform((m, 3))
        lo, hi = np.log(C.AUG_FAKTOR[0]), np.log(C.AUG_FAKTOR[1])
        faktor = np.exp(lo + (hi - lo) * u[:, 0])
        v = C.AUG_VERSCHIEBUNG[0] + (C.AUG_VERSCHIEBUNG[1] - C.AUG_VERSCHIEBUNG[0]) * u[:, 1:3]
        return faktor[:, None, None] * X + v[:, None, :]
    raise ValueError(f"unbekannte Eingabe {art!r}")


def train_zeiger(eingabe: str, init_seed: int, steps: int = C.SCHRITTE_ZEIGER, batch: int = C.BATCH, d: int = C.D_ZEIGER, lr: float = C.LERNRATE,
                 groessen: tuple = (C.N_TRAIN,), progress=None, progress_every: int = 250, max_seconds: float | None = None):
    """Trainiert ein Zeigernetz (eine Schicht, Attention und Vorwärtsnetz, wie A) mit REINFORCE und Referenzwert „Mittel“. Gewichte, Trainingsinstanzen und Ziehungen
    kommen aus denselben Strömen wie in `nfi_train.train` (gepaart mit A: gleicher `init_seed` = `data_seed`); bei mehreren `groessen` zieht jede Charge eine Größe
    (Strom `ZWECK_MISCH`). Rückgabe (P, counts)."""
    P = N.init_params(d, UStream(SC.derive(C.ZWECK_INIT, d, N.variant_code(1, True, True), init_seed)), layers=1, attn=True, ffn=True)
    opt = T.Adam(P, lr)
    gemischt = len(groessen) > 1
    xs = UStream(SC.derive(C.ZWECK_TRAIN_X, 0 if gemischt else groessen[0], init_seed))
    ss = UStream(SC.derive(C.ZWECK_TRAIN_S, 0 if gemischt else groessen[0], init_seed))
    ug = UStream(SC.derive(C.ZWECK_AUG, init_seed)) if eingabe == "augmentiert" else None
    um = UStream(SC.derive(C.ZWECK_MISCH, init_seed)) if gemischt else None
    starts = np.zeros((batch, 1), dtype=np.int64)
    t0 = time.perf_counter()
    touren = 0
    erledigt = 0
    for step in range(1, steps + 1):
        n = groessen[int(um.uniform((1,))[0] * len(groessen))] if gemischt else groessen[0]
        X = I.train_batch(xs, batch, n)
        Xin = eingabe_veraendern(X, eingabe, ug)
        H, ec = N.encode(P, Xin)
        out = N.decode(P, H, starts, mode="sample", u_stream=ss)
        L = N.tour_lengths(X, out["tours"])
        w = (L - float(L.mean())) / L.size
        Gd, dH = N.decode_backward(P, out["cache"], w)
        opt.step(P, {**N.encode_backward(P, ec, dH), **Gd})
        touren += batch
        erledigt = step
        if progress is not None and step % progress_every == 0:
            progress(step, steps)
        if max_seconds is not None and time.perf_counter() - t0 > max_seconds:
            break
    return P, {"schritte": erledigt, "touren": touren, "sekunden": time.perf_counter() - t0}


def train_merkmal(skala: str, seed: int, steps: int = KC.K_SCHRITTE_FROZEN, batch_inst: int = KC.K_INST, rollouts: int = KC.K_ROLL, lr: float = KC.K_LERNRATE,
                  n: int = KC.K_N_TRAIN, s0: float = C.S0, progress=None, progress_every: int = 250, max_seconds: float | None = None):
    """Trainiert das Merkmalsnetz K. `skala`: "mit" (s je Instanz) oder "ohne" (feste Skala `s0`). Gleiche Ströme wie `nfik_policy.train_featnet` (Gewichte, Instanzen,
    Startstädte, Gumbel-Ziehungen); mit "mit" ist die Schleife schrittgleich zu ihr. Rückgabe (Parameter, counts)."""
    params = MLP.init_mlp(KC.K_MERKMALE, KC.H, UStream(SC.derive(KC.ZWECK_K_INIT, seed)))
    opt = MLP.Adam(params, lr)
    us_i = UStream(SC.derive(KC.ZWECK_K_INST, seed))
    us_s = UStream(SC.derive(KC.ZWECK_K_START, seed))
    us_g = UStream(SC.derive(KC.ZWECK_K_ZIEHEN, seed))
    diag = np.diag(np.full(n, np.inf))
    t0 = time.perf_counter()
    zeilen = 0
    erledigt = 0
    for step in range(steps):
        Ds = I.distance_matrix_np(I.train_batch(us_i, batch_inst, n))
        if skala == "mit":
            ss = (Ds + diag).min(2).mean(1)
        elif skala == "ohne":
            ss = np.full(batch_inst, s0)
        else:
            raise ValueError(f"unbekannte Skala {skala!r}")
        starts = us_s.choice(batch_inst, n, rollouts).ravel()
        info = KP.reinforce_schritt(params, opt, np.repeat(Ds, rollouts, axis=0), np.repeat(ss, rollouts), starts, us_g, rollouts)
        zeilen += info["zeilen"]
        erledigt = step + 1
        if progress is not None and erledigt % progress_every == 0:
            progress(erledigt, steps)
        if max_seconds is not None and time.perf_counter() - t0 > max_seconds:
            break
    return params, {"schritte": erledigt, "touren": erledigt * batch_inst * rollouts, "zeilen": zeilen, "sekunden": time.perf_counter() - t0}


def feature_rollout_greedy(params, D: np.ndarray, s: np.ndarray, starts: np.ndarray):
    """Greedy-Touren des Merkmalsnetzes für R Instanzen D (R, n, n) mit Skalen s (R,) ab starts (R,): (Touren, Längen, Zeilen)."""
    T_, L, zeilen, _ = KF.rollout(D, s, starts, params, mode="greedy")
    return T_, L, zeilen

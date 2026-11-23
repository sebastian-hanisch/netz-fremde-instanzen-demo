"""Trainiert die Netze dieser Demo, je Lauf eine Datei in `nfi_models/` (setzt fort: vorhandene Dateien werden übersprungen).

Aufruf (Wurzelordner der Demo, Test-venv, BLAS-Threads 1):
    python tools/nfi_train_models.py --netze Bohne An            # nur diese Netze
    python tools/nfi_train_models.py --netze Aaug Amix --seeds 0 1 2
    python tools/nfi_train_models.py --s0                         # prüft die feste Skala s0 an 20 000 Trainingsinstanzen

Netze: Bohne (Merkmalsnetz mit fester Skala s0), An (Zeigernetz, Eingabe auch im Training normiert), Aaug (im Training skaliert und verschoben), Amix (normiert,
Größe je Schritt aus {10, 20, 50}). A und B mit s liegen als Kopien aus Stück 3 und 4 in `nfi_models/` (`tools/nfi_kopieren.py`). Rezepte: Zeigernetze 8 000 Schritte x 120 Touren
(960 000 Touren, Lernrate 0,001, Gradientennorm 1, Referenzwert Mittel der Charge), Merkmalsnetz 3 750 Schritte x 256 Touren (960 000 Touren, Lernrate 0,03, Gradientennorm 5, Referenzwert
Mittel der 8 Touren je Instanz). Gewichts- und Daten-Zufalls-Seed gepaart (0 bis 4, wie A beziehungsweise B mit s). Das Protokoll steht in `tools/_zwischen/train_log.txt`."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import nfi_constants as C  # noqa: E402
import nfi_instances as I  # noqa: E402
import nfi_net as N  # noqa: E402
import nfi_scenario as SC  # noqa: E402
import nfi_trainloops as TL  # noqa: E402
import nfi_transform as TR  # noqa: E402
import nfik_constants as KC  # noqa: E402
import nfik_mlp as MLP  # noqa: E402
from nfi_rngvec import UStream  # noqa: E402

ZWISCHEN = ROOT / "tools" / "_zwischen"
VAL_ANZAHL = 100


def protokoll(text: str) -> None:
    ZWISCHEN.mkdir(exist_ok=True)
    zeile = f"{time.strftime('%H:%M:%S')} {text}"
    print(zeile, flush=True)
    with open(ZWISCHEN / "train_log.txt", "a", encoding="utf-8") as f:
        f.write(zeile + "\n")


def pruefe_s0() -> float:
    us = UStream(SC.derive(C.ZWECK_TRAIN_X, 20, 99))
    X = I.train_batch(us, 20000, 20)
    s = TR.skala_s(I.distance_matrix_np(X))
    return float(s.mean())


def val_menge():
    X = I.val_set(20, VAL_ANZAHL, "gleich")
    D = I.distance_matrix_np(X)
    nn = TR.tour_laengen(D, TR.nn_touren(D))
    return X, D, nn


def val_zeiger(P, X, nn, normiert: bool) -> float:
    Xin = I.normalize_unit(X) if normiert else X
    starts = np.zeros((len(X), 1), dtype=np.int64)
    T_ = N.greedy_tours(P, Xin, starts)
    L = N.tour_lengths(X, T_)[:, 0]
    return float(np.mean((L - nn) / nn) * 100.0)


def val_merkmal(params, D, nn, skala: str) -> float:
    s = TR.skala_s(D) if skala == "mit" else np.full(len(D), C.S0)
    _, L, _ = TL.feature_rollout_greedy(params, D, s, np.zeros(len(D), dtype=np.int64))
    return float(np.mean((L - nn) / nn) * 100.0)


EINGABE_DES_NETZES = {"An": "normiert", "Aaug": "augmentiert", "Amix": "normiert"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--netze", nargs="*", default=["Bohne", "An"])
    ap.add_argument("--seeds", nargs="*", type=int, default=list(range(C.NETZ_SEEDS)))
    ap.add_argument("--steps", type=int, default=None, help="nur für Proben; Standard: volles Rezept")
    ap.add_argument("--s0", action="store_true")
    a = ap.parse_args()
    if a.s0:
        m = pruefe_s0()
        print(f"mittleres s über 20 000 Trainingsinstanzen (n = 20): {m:.6f}; Konstante S0 = {C.S0:.6f}; Abstand {abs(m - C.S0):.6f}")
        return 0
    X, D, nn = val_menge()
    (ROOT / C.MODELL_ORDNER).mkdir(exist_ok=True)
    for netz in a.netze:
        for seed in a.seeds:
            pfad = ROOT / C.MODELL_ORDNER / f"nfi_{netz}_s{seed}.npz"
            if pfad.exists():
                protokoll(f"{netz} s{seed}: vorhanden, übersprungen")
                continue

            def fortschritt(k, g, _netz=netz, _seed=seed):
                protokoll(f"{_netz} s{_seed}: Schritt {k} von {g}")

            if netz == "Bohne":
                steps = a.steps or C.SCHRITTE_MERKMAL
                params, cnt = TL.train_merkmal("ohne", seed, steps=steps, progress=fortschritt, progress_every=500)
                meta = {"kind": "K", "seed": seed, "steps": cnt["schritte"], "lr": KC.K_LERNRATE, "n_train_tours": cnt["touren"], "n_train_rows": cnt["zeilen"],
                        "feature_version": 1, "n": 20, "s_mode": "ohne", "s0": C.S0, "val_unterschied_zu_nn": val_merkmal(params, D, nn, "ohne"), "val_instanzen": VAL_ANZAHL}
                MLP.save_model(pfad, params, meta)
                protokoll(f"{netz} s{seed}: fertig, {cnt['schritte']} Schritte, {cnt['touren']} Touren, {cnt['sekunden']:.0f} s, Validierung (Unterschied zu NN) {meta['val_unterschied_zu_nn']:.2f} Prozentpunkte")
            elif netz in EINGABE_DES_NETZES:
                steps = a.steps or C.SCHRITTE_ZEIGER
                groessen = C.MISCH_GROESSEN if netz == "Amix" else (C.N_TRAIN,)
                P, cnt = TL.train_zeiger(EINGABE_DES_NETZES[netz], seed, steps=steps, groessen=groessen, progress=fortschritt, progress_every=1000)
                meta = {"n_train": 20 if netz != "Amix" else list(groessen), "steps": cnt["schritte"], "batch": C.BATCH, "lr": C.LERNRATE, "baseline": "mean", "init_seed": seed,
                        "data_seed": seed, "variante": netz, "erzeugt_von": "tools/nfi_train_models.py", "eingabe_im_training": EINGABE_DES_NETZES[netz],
                        "touren": cnt["touren"], "val_unterschied_zu_nn": val_zeiger(P, X, nn, EINGABE_DES_NETZES[netz] == "normiert"), "val_instanzen": VAL_ANZAHL}
                N.save_model(pfad, P, meta)
                protokoll(f"{netz} s{seed}: fertig, {cnt['schritte']} Schritte, {cnt['touren']} Touren, {cnt['sekunden']:.0f} s, Validierung (Unterschied zu NN) {meta['val_unterschied_zu_nn']:.2f} Prozentpunkte")
            else:
                raise SystemExit(f"unbekanntes Netz {netz}")
    (ZWISCHEN / "train_ende.json").write_text(json.dumps({"netze": a.netze, "seeds": a.seeds}), encoding="utf-8")
    protokoll("Ende")
    return 0


if __name__ == "__main__":
    sys.exit(main())

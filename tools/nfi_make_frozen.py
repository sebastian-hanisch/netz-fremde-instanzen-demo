"""Erzeugt die eingefrorene Messung `nfi_frozen.json` (Format `nfi-frozen-1`) aus den vorgerechneten Netzen (`nfi_models/`) und den Referenzen (`nfi_refs.json`).

Aufruf (Wurzelordner der Demo, Test-venv, BLAS-Threads 1), Stufen setzen fort (Zwischenstände in `tools/_zwischen/`):
    python tools/nfi_make_frozen.py --stage base                      # Basiszellen: 3 Familien x 5 Größen, alle Netze, die vorliegen
    python tools/nfi_make_frozen.py --stage base --netze Aaug Amix    # ergänzt die Netze des Mischtrainings
    python tools/nfi_make_frozen.py --stage wechsel                   # Skala- und Lagewechsel: 3 Familien x n = 20, 50 x 3 Wechsel
    python tools/nfi_make_frozen.py --stage diagnose                  # Sättigung, Kappung, s/s0
    python tools/nfi_make_frozen.py --stage final                     # schreibt nfi_frozen.json
    python tools/nfi_make_frozen.py --check                           # rechnet einen Ausschnitt neu und vergleicht als Band

Einheit überall: **eine greedy-Tour je Instanz, Start Stadt 0** (K = n(n-1)/2 für Netz und Nächster Nachbar). Längen im Einheitsquadrat, gemessen auf der Basisinstanz."""

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
import nfi_diagnose as DG  # noqa: E402
import nfi_eval as EV  # noqa: E402
import nfi_gegner as GG  # noqa: E402
import nfi_instances as I  # noqa: E402
import nfi_models as MO  # noqa: E402
import nfi_transform as TR  # noqa: E402

ZWISCHEN = ROOT / "tools" / "_zwischen"
RUNDEN = 7
FORMAT = "nfi-frozen-1"


def _lade(f: Path) -> dict:
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _speichere(f: Path, d: dict) -> None:
    f.write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")


def runden(a) -> list:
    return np.round(np.asarray(a, dtype=np.float64), RUNDEN).tolist()


def stufe_base(netze: list[str]) -> None:
    for n in C.GROESSEN:
        for fam in C.FAMILIEN:
            f = ZWISCHEN / f"base_{fam}_{n}.json"
            z = _lade(f)
            m = C.INSTANZEN[n]
            X = TR.testmenge(fam, n, m)
            D = TR.abstaende(X)
            t0 = time.time()
            if "nn" not in z:
                T = TR.nn_touren(D)
                assert TR.ist_permutation(T)
                z["nn"] = runden(TR.tour_laengen(D, T))
                m2 = m if n < 200 else min(m, C.NN2OPT_INSTANZEN_GROSS)
                laengen2, ks = [], []
                for i in range(m2):
                    Dl = I.distance_lists(X[i])
                    tour, k_nn = GG.nn_tour_k(Dl, 0)
                    _, L2, k2 = GG.two_opt_k(Dl, tour)
                    laengen2.append(L2)
                    ks.append(k_nn + k2)
                z["nn2opt"] = runden(laengen2)
                z["k_nn2opt_mittel"] = float(np.mean(ks))
            z.setdefault("modelle", {})
            for netz in netze:
                if netz in z["modelle"]:
                    continue
                if not MO.vorhanden(netz):
                    print(f"  {netz}: Modelle fehlen, übersprungen", flush=True)
                    continue
                L, T = EV.laengen_netz(netz, MO.lade_alle(netz), X, D)
                z["modelle"][netz] = runden(L)
            _speichere(f, z)
            print(f"base {fam} n = {n} (m = {m}): {time.time() - t0:.0f} s, Netze {sorted(z['modelle'])}", flush=True)


def stufe_wechsel(netze: list[str]) -> None:
    for n in C.WECHSEL_GROESSEN:
        for fam in C.FAMILIEN:
            m = C.INSTANZEN[n]
            X = TR.testmenge(fam, n, m)
            D = TR.abstaende(X)
            for art in ("klein", "gross", "versch"):
                f = ZWISCHEN / f"wechsel_{fam}_{n}_{art}.json"
                z = _lade(f)
                z.setdefault("modelle", {})
                z.setdefault("invarianz", {})
                t0 = time.time()
                Xw = TR.wechsel(X, art)
                for netz in netze:
                    if netz in z["modelle"] or netz in z["invarianz"] or not MO.vorhanden(netz):
                        continue
                    modelle = MO.lade_alle(netz)
                    L, T = EV.laengen_netz(netz, modelle, Xw, D)
                    if netz in C.WECHSEL_NETZE:
                        z["modelle"][netz] = runden(L)
                    else:                                  # invariante Netze: nur der Anteil gleicher Touren wie auf der Basisinstanz
                        _, T0 = EV.laengen_netz(netz, modelle, X, D)
                        z["invarianz"][netz] = float((T == T0).all(-1).mean())
                _speichere(f, z)
                print(f"wechsel {fam} n = {n} {art}: {time.time() - t0:.0f} s, Netze {sorted(z['modelle'])}, invariant {z['invarianz']}", flush=True)


def stufe_diagnose() -> None:
    f = ZWISCHEN / "diagnose.json"
    out = {"saettigung": {}, "kappung_bohne": {}, "kappung_bmit": {}, "s_verhaeltnis": {}, "instanzen": C.DIAGNOSE_INSTANZEN}
    m = C.DIAGNOSE_INSTANZEN
    X = TR.testmenge("gleich", 20, m)
    for art in C.WECHSEL:
        Xw = TR.wechsel(X, art)
        je_seed = [DG.saettigung_anteil(P, Xw) for P in MO.lade_alle("A")]
        out["saettigung"][art] = {"mittel": float(np.mean(je_seed)), "je_seed": je_seed}
        D = TR.abstaende(Xw)
        s = np.full(len(D), C.S0)
        ks = [DG.kappung_anteil(p, D, s) for p in MO.lade_alle("Bohne")]
        out["kappung_bohne"][art] = {k: float(np.mean([x[k] for x in ks])) for k in ks[0]}
    D = TR.abstaende(X)
    ks = [DG.kappung_anteil(p, D, TR.skala_s(D)) for p in MO.lade_alle("Bmit")]
    out["kappung_bmit"]["basis"] = {k: float(np.mean([x[k] for x in ks])) for k in ks[0]}
    for n in C.GROESSEN:
        for fam in C.FAMILIEN:
            Xn = TR.testmenge(fam, n, C.INSTANZEN[n])
            out["s_verhaeltnis"][f"{fam}_{n}"] = DG.s_verhaeltnis(TR.abstaende(Xn))
    _speichere(f, out)
    print("diagnose fertig", flush=True)


def stufe_final() -> None:
    refs = json.loads((ROOT / "nfi_refs.json").read_text(encoding="utf-8"))
    out = {"format": FORMAT, "meta": {}, "instanzen": {str(n): C.INSTANZEN[n] for n in C.GROESSEN}, "base": {}, "wechsel": {}}
    netze_basis = set()
    for n in C.GROESSEN:
        for fam in C.FAMILIEN:
            z = _lade(ZWISCHEN / f"base_{fam}_{n}.json")
            ref_basis = np.array(refs["ref"][f"{fam}_{n}"])
            assert len(ref_basis) == len(z["nn"]), (fam, n)
            kandidaten = [np.array(z["nn"])[None]] + [np.array(v) for v in z["modelle"].values()]
            if n in C.WECHSEL_GROESSEN:                     # auch die Touren unter den Wechseln sind Touren der Basisinstanz
                for art in ("klein", "gross", "versch"):
                    w = _lade(ZWISCHEN / f"wechsel_{fam}_{n}_{art}.json")
                    kandidaten += [np.array(v) for v in w.get("modelle", {}).values()]
            gebaut = np.concatenate(kandidaten).min(0)
            if "nn2opt" in z:
                m2 = len(z["nn2opt"])
                gebaut[:m2] = np.minimum(gebaut[:m2], np.array(z["nn2opt"]))
            if n <= 20:
                ref = ref_basis
                unterboten = 0.0
            else:
                ref = np.minimum(ref_basis, gebaut)
                unterboten = float((gebaut < ref_basis - 1e-9).mean())
            z["ref"] = runden(ref)
            z["ref_basis"] = runden(ref_basis)
            z["unterboten_anteil"] = unterboten
            netze_basis |= set(z["modelle"])
            out["base"][f"{fam}_{n}"] = z
    for n in C.WECHSEL_GROESSEN:
        for fam in C.FAMILIEN:
            for art in ("klein", "gross", "versch"):
                out["wechsel"][f"{fam}_{n}_{art}"] = _lade(ZWISCHEN / f"wechsel_{fam}_{n}_{art}.json")
    out["diagnose"] = _lade(ZWISCHEN / "diagnose.json")
    out["meta"] = {
        "format": FORMAT, "netze": sorted(netze_basis), "seeds": C.NETZ_SEEDS, "s0": C.S0, "instanzen": out["instanzen"], "wechsel": {"klein": C.FAKTOR_KLEIN, "gross": C.FAKTOR_GROSS, "versch": list(C.VERSCHIEBUNG)},
        "referenz": refs["art"], "numpy": np.__version__, "modelle": MO.modelle_info(),
    }
    _speichere(ROOT / "nfi_frozen.json", out)
    print("nfi_frozen.json geschrieben,", (ROOT / "nfi_frozen.json").stat().st_size, "Bytes", flush=True)


def stufe_check() -> int:
    """Rechnet drei Instanzen je Zelle (n = 20, 50, Basis und drei Wechsel) neu und vergleicht das Zellenmittel mit der eingefrorenen Messung (Band: höchstens 0,5 Prozentpunkte)."""
    fr = json.loads((ROOT / "nfi_frozen.json").read_text(encoding="utf-8"))
    fehler = 0
    for n in (20, 50):
        for fam in C.FAMILIEN:
            X = TR.testmenge(fam, n, 3)
            D = TR.abstaende(X)
            for art in C.WECHSEL:
                Xw = TR.wechsel(X, art)
                for netz in ("A", "Bohne") if art != "basis" else ("A", "Ant", "An", "Bmit", "Bohne"):
                    if netz not in (fr["base"][f"{fam}_{n}"]["modelle"] if art == "basis" else fr["wechsel"][f"{fam}_{n}_{art}"]["modelle"]):
                        continue
                    L, _ = EV.laengen_netz(netz, MO.lade_alle(netz), Xw, D)
                    alt = np.array((fr["base"][f"{fam}_{n}"]["modelle"] if art == "basis" else fr["wechsel"][f"{fam}_{n}_{art}"]["modelle"])[netz])[:, :3]
                    abw = float(np.abs(100 * (L.mean() - alt.mean()) / alt.mean()))
                    if abw > 0.5:
                        print("ABWEICHUNG", fam, n, art, netz, abw)
                        fehler += 1
    print("check:", "ok" if not fehler else f"{fehler} Abweichungen")
    return 1 if fehler else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("base", "wechsel", "diagnose", "final"), default=None)
    ap.add_argument("--netze", nargs="*", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ZWISCHEN.mkdir(exist_ok=True)
    if a.check:
        return stufe_check()
    if a.stage == "base":
        stufe_base(a.netze or list(C.NETZE))
    elif a.stage == "wechsel":
        stufe_wechsel(a.netze or ["A", "Bohne", "Aaug", "Ant", "An", "Bmit", "Amix"])
    elif a.stage == "diagnose":
        stufe_diagnose()
    elif a.stage == "final":
        stufe_final()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Referenzen für die Instanzen der eingefrorenen Messung: `nfi_refs.json`.

n = 10 und 20 exakt (Held-Karp, `nfi_exact.held_karp`); n = 50, 100, 200 die beste Lösung der Metaheuristik „Guided Local Search“ von OR-Tools (nur hier und in den Tests, nie in der App)
mit festem Zeitlimit je Instanz. „Beste bekannte Lösung“ in der Auswertung ist das Minimum aus dieser Lösung und allen gebauten Touren (`tools/nfi_make_frozen.py`); die Lücken sind dort
Untergrenzen, die gepaarten Unterschiede dagegen kaum betroffen (die Referenz steht nur im Nenner, die zu große Referenz staucht sie nur leicht).

Aufruf (Wurzelordner, Test-venv, BLAS-Threads 1):
    python tools/nfi_make_refs.py --groessen 10 20            # exakt
    python tools/nfi_make_refs.py --groessen 50 100 200       # GLS (Zeitlimit nfi_constants.GLS_SEKUNDEN je Größe)
Zwischenstände: tools/_zwischen/refs_<familie>_<n>.json (setzt fort); `--final` schreibt `nfi_refs.json`."""

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
import nfi_exact as EX  # noqa: E402
import nfi_transform as TR  # noqa: E402

ZWISCHEN = ROOT / "tools" / "_zwischen"
SKALIERUNG = 10 ** 6


def gls_laenge(X: np.ndarray, sekunden: float) -> tuple[float, list]:
    """Beste Tour, die Guided Local Search von OR-Tools in `sekunden` findet (Abstände ganzzahlig auf 10^6 skaliert); Länge in Gleitkomma nachgerechnet."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    D = TR.abstaende(X[None])[0]
    Di = np.rint(D * SKALIERUNG).astype(np.int64).tolist()
    n = len(X)
    mgr = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(mgr)
    cb = routing.RegisterTransitCallback(lambda i, j: Di[mgr.IndexToNode(i)][mgr.IndexToNode(j)])
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromMilliseconds(int(sekunden * 1000))
    lsg = routing.SolveWithParameters(p)
    index = routing.Start(0)
    tour = []
    while not routing.IsEnd(index):
        tour.append(mgr.IndexToNode(index))
        index = lsg.Value(routing.NextVar(index))
    t = np.array(tour)
    return float(TR.tour_laengen(D[None], t[None])[0]), tour


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groessen", nargs="*", type=int, default=[10, 20])
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    ZWISCHEN.mkdir(exist_ok=True)
    if a.final:
        out = {"format": "nfi-refs-1", "art": {}, "gls_sekunden": C.GLS_SEKUNDEN, "ref": {}}
        for n in C.GROESSEN:
            out["art"][str(n)] = "exakt (Held-Karp)" if n <= 20 else f"Guided Local Search (OR-Tools), {C.GLS_SEKUNDEN[n]} s je Instanz"
            for fam in C.FAMILIEN:
                f = ZWISCHEN / f"refs_{fam}_{n}.json"
                if not f.exists():
                    print("fehlt:", f.name)
                    return 1
                out["ref"][f"{fam}_{n}"] = json.loads(f.read_text(encoding="utf-8"))["laengen"]
        (ROOT / "nfi_refs.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
        print("nfi_refs.json geschrieben")
        return 0
    for n in a.groessen:
        for fam in C.FAMILIEN:
            f = ZWISCHEN / f"refs_{fam}_{n}.json"
            if f.exists():
                print("vorhanden:", f.name, flush=True)
                continue
            m = C.INSTANZEN[n]
            X = TR.testmenge(fam, n, m)
            t0 = time.time()
            if n <= 20:
                D = TR.abstaende(X)
                laengen = [float(EX.held_karp(D[i])) for i in range(m)]
            else:
                laengen = [gls_laenge(X[i], C.GLS_SEKUNDEN[n])[0] for i in range(m)]
            f.write_text(json.dumps({"familie": fam, "n": n, "m": m, "laengen": [round(x, 9) for x in laengen], "sekunden": time.time() - t0}), encoding="utf-8")
            print(f"{fam} n = {n}: {m} Instanzen in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

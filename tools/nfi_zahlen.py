"""Druckt alle Kennzahlen der eingefrorenen Messung für die README (`nfi_befunde.befunde`) und die Zellen der Tabellen.

Aufruf (Wurzelordner, Test-venv):  python tools/nfi_zahlen.py [--zellen]"""

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nfi_befunde as BF  # noqa: E402
import nfi_constants as C  # noqa: E402
import nfi_evaluation as V  # noqa: E402
import nfi_experiments as EX  # noqa: E402


def main() -> int:
    fr = EX.Frozen.laden()
    b = BF.befunde(fr)
    print(json.dumps(b, indent=1, ensure_ascii=False, default=str))
    if "--zellen" in sys.argv:
        for n in C.GROESSEN:
            for fam in C.FAMILIEN:
                zs = []
                for netz in fr.netze():
                    z = fr.zelle(fam, n, netz)
                    zs.append(f"{C.NETZ_KURZ[netz]} {V.de(z['luecke'], 2)} / {V.de_vz(z['v'], 2)} {V.intervall(z['lo'], z['hi'], 2)} {z['seeds_pos']}/5 {z['klasse']}")
                print(f"n = {n} {fam} (m = {fr.m(n)}): NN {V.de(fr.luecke_nn(fam, n), 2)}; 2-opt {fr.luecke_nn2opt(fam, n)}; " + "; ".join(zs))
        for n in C.WECHSEL_GROESSEN:
            for fam in C.FAMILIEN:
                for art in ("klein", "gross", "versch"):
                    zs = []
                    for netz in ("A", "Bohne", "Aaug"):
                        z = fr.zelle(fam, n, netz, art)
                        if z:
                            zs.append(f"{C.NETZ_KURZ[netz]} {V.de(z['luecke'], 2)} / {V.de_vz(z['v'], 2)} {V.intervall(z['lo'], z['hi'], 2)} {z['seeds_pos']}/5 {z['klasse']}")
                    print(f"wechsel n = {n} {fam} {art}: " + "; ".join(zs))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Schreibt `nfi_models/MODELLE.json`: Größe, Zufalls-Seeds, Schritte, Touren und Prüfsumme (sha256) jeder vorgerechneten Modelldatei.

Aufruf (Wurzelordner, Test-venv):  python tools/nfi_modelle_info.py"""

import hashlib
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nfi_constants as C  # noqa: E402
import nfi_models as MO  # noqa: E402

BESCHREIBUNG = {
    "A": "Zeigernetz, rohe Koordinaten als Eingabe, bei n = 20 gleichverteilt trainiert (Kopie aus rechnen-beim-einsatz-demo, Rezept von zeigernetz-demo)",
    "An": "Zeigernetz, Eingabe auch im Training auf das Einheitsquadrat normiert (gleiche Gewichts- und Daten-Zufalls-Seeds wie A)",
    "Aaug": "Zeigernetz, rohe Eingabe, im Training je Instanz skaliert (Faktor 0,1 bis 10, logarithmisch gleichverteilt) und verschoben (-3 bis 3)",
    "Amix": "Zeigernetz, normierte Eingabe, im Training je Schritt n = 10, 20 oder 50",
    "Bmit": "Merkmalsnetz K mit der Skala s der Instanz (Kopie aus zugwahl-lernen-demo)",
    "Bohne": "Merkmalsnetz K mit der festen Skala s0 (gleiche Ströme und Gewichts-Zufalls-Seeds wie Bmit)",
}


def main() -> int:
    out = {"format": "nfi-modelle-1", "s0": C.S0, "netze": {}}
    for netz in ("A", "An", "Aaug", "Amix", "Bmit", "Bohne"):
        dateien = []
        for s in range(C.NETZ_SEEDS):
            p = MO.pfad(netz, s)
            if not p.exists():
                continue
            P, meta = MO.lade(netz, s)
            dateien.append({
                "datei": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "zufalls_seed": s,
                "schritte": int(meta.get("steps", 0)), "touren": int(meta.get("touren", meta.get("n_train_tours", int(meta.get("steps", 0)) * C.BATCH))),
                "val_unterschied_zu_nn": (float(meta["val_unterschied_zu_nn"]) if "val_unterschied_zu_nn" in meta else None),
            })
        if dateien:
            out["netze"][netz] = {"beschreibung": BESCHREIBUNG[netz], "dateien": dateien}
    (ROOT / C.MODELL_ORDNER / "MODELLE.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("MODELLE.json:", {k: len(v["dateien"]) for k, v in out["netze"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())

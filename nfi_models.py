"""Vorgerechnete Netze laden: Zeigernetz (Format znz-modell-1) und Merkmalsnetz K (Format zwl-mlp-1), je fünf Gewichts-Zufalls-Seeds.

Dateien in `nfi_models/`: `nfi_<Netz>_s<Seed>.npz` mit Netz in A (Zeigernetz, rohe Eingabe; Kopie aus Stück 3), An (Eingabe auch im Training normiert), Aaug (augmentiert), Amix (Mischtraining)
und Bmit (Merkmalsnetz mit s; Kopie aus Stück 4), Bohne (Merkmalsnetz mit fester Skala s0). Das Netz „A+nt“ (`Ant`) sind die Gewichte von A mit normierter Eingabe beim Einsatz.
`nfi_models/MODELLE.json` führt Größe, Zufalls-Seeds und Prüfsummen. Trainiert wird nur in `tools/` (`nfi_trainloops.py`), nie in der App."""

import json
from pathlib import Path

import nfi_constants as C
import nfi_net as N
import nfik_mlp as MLP

ORDNER = Path(__file__).resolve().parent / C.MODELL_ORDNER
DATEI_NETZ = {"A": "A", "Ant": "A", "An": "An", "Bmit": "Bmit", "Bohne": "Bohne", "Aaug": "Aaug", "Amix": "Amix"}   # Gewichte-Datei je Netz (A+nt benutzt die Gewichte von A)


def pfad(netz: str, seed: int) -> Path:
    return ORDNER / f"nfi_{DATEI_NETZ[netz]}_s{int(seed)}.npz"


def ist_zeiger(netz: str) -> bool:
    return netz in C.ZEIGER_NETZE


def vorhanden(netz: str) -> bool:
    """Liegen alle Seeds des Netzes als Datei vor?"""
    return all(pfad(netz, s).exists() for s in range(C.NETZ_SEEDS))


def lade(netz: str, seed: int):
    """(Parameter, Angaben) eines Modells; Zeigernetz: Wörterbuch P, Merkmalsnetz: Liste [W1, b1, W2, b2]. Fehlerhafte Dateien lösen ValueError aus."""
    p = pfad(netz, seed)
    return N.load_model(p) if ist_zeiger(netz) else MLP.load_model(p)


def lade_alle(netz: str, seeds=None):
    """Liste der Parameter über die Seeds (Standard alle fünf)."""
    seeds = range(C.NETZ_SEEDS) if seeds is None else seeds
    return [lade(netz, s)[0] for s in seeds]


def modelle_info() -> dict:
    """Inhalt von `MODELLE.json` oder ein leeres Wörterbuch."""
    f = ORDNER / "MODELLE.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}

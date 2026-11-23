"""Erzeugt die Kopien aus den Quellen (zeigernetz-demo, Präfix znz_ -> nfi_; zugwahl-lernen-demo, Präfix zwl_ -> nfik_/nfi_) und die Modelle aus den Quellen.

Aufruf (Wurzelordner der Demo):  python tools/nfi_kopieren.py            # schreibt alle Kopien neu
                                  python tools/nfi_kopieren.py --check    # meldet Abweichungen, schreibt nichts

Zwei Präfixe wie in `rechnen-beim-einsatz-demo` (rbe_, rbek_), weil beide Quellen eine `*_constants.py` haben und die Modellformate verschieden heißen
(`znz-modell-1` für das Zeigernetz, `zwl-mlp-1` für das Merkmalsnetz K): `nfi_` führt die Dateien aus Stück 1 (Zufall, Instanzen, Zeigernetz, exakte Wege, Zählung, Gegner),
`nfik_` die aus Stück 4 (Netz 7-16-1, Merkmale, Politikgradient). Die Politikgradient-Datei der Quelle importiert `zwl_instances`, `zwl_scenario` und `zwl_rngvec`; das sind in
Stück 4 selbst Kopien der Dateien aus Stück 1, hier heißen sie darum `nfi_instances`, `nfi_scenario`, `nfi_rngvec` (Ersetzungsregel in `ERSETZUNG_ZWL`, vor dem allgemeinen
`zwl_` -> `nfik_`). Die Konstantendateien `nfi_constants.py` und `nfik_constants.py` sind keine Kopien: sie führen dieselben Namen und Werte (Test `test_copies.py`).
Zeilenenden werden nicht verändert; der Kopier-Test vergleicht nach Umwandlung von CRLF in LF."""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJEKTE = ROOT.parent
QUELLE_1 = PROJEKTE / "zeigernetz-demo"
QUELLE_3 = PROJEKTE / "rechnen-beim-einsatz-demo"
QUELLE_4 = PROJEKTE / "zugwahl-lernen-demo"
VORLAGE_RNG = PROJEKTE / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"

AUS_STUECK_1 = ["rngblock", "rngvec", "scenario", "instances", "exact", "stats", "net", "train", "cost", "gegner"]
AUS_STUECK_4 = ["mlp", "featnet", "policy"]
ERSETZUNG_ZWL = [(b"zwl_instances", b"nfi_instances"), (b"zwl_scenario", b"nfi_scenario"), (b"zwl_rngvec", b"nfi_rngvec"), (b"zwl_", b"nfik_")]
MODELLE_K = [f"featnet_s{k}.npz" for k in range(5)]
MODELLE_A = [f"rbe_n20_voll_d32_s{k}.npz" for k in range(5)]


def _lf(b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n")


def kopie_znz(quelle: Path) -> bytes:
    return _lf(quelle.read_bytes()).replace(b"znz_", b"nfi_")


def kopie_zwl(quelle: Path) -> bytes:
    inhalt = _lf(quelle.read_bytes())
    for alt, neu in ERSETZUNG_ZWL:
        inhalt = inhalt.replace(alt, neu)
    return inhalt


def aufgaben():
    """(Ziel, Bytes) aller Textkopien."""
    out = [(ROOT / "nfi_rng.py", _lf(VORLAGE_RNG.read_bytes())), (ROOT / "modul_frisch.py", _lf((QUELLE_1 / "modul_frisch.py").read_bytes()))]
    for n in AUS_STUECK_1:
        out.append((ROOT / f"nfi_{n}.py", kopie_znz(QUELLE_1 / f"znz_{n}.py")))
    for n in AUS_STUECK_4:
        out.append((ROOT / f"nfik_{n}.py", kopie_zwl(QUELLE_4 / f"zwl_{n}.py")))
    return out


def modell_aufgaben():
    """(Ziel, Quelle) der Modelldateien, die unverändert (Bytekopie) übernommen werden: Zeigernetz A (Seeds 0 bis 4, aus Stück 3) und Merkmalsnetz mit s (aus Stück 4)."""
    out = []
    for k in range(5):
        out.append((ROOT / "nfi_models" / f"nfi_A_s{k}.npz", QUELLE_3 / "rbe_models" / MODELLE_A[k]))
        out.append((ROOT / "nfi_models" / f"nfi_Bmit_s{k}.npz", QUELLE_4 / "zwl_modelle" / MODELLE_K[k]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    fehler = 0
    for ziel, inhalt in aufgaben():
        if a.check:
            if not ziel.exists() or _lf(ziel.read_bytes()) != inhalt:
                print("ABWEICHUNG:", ziel.name)
                fehler += 1
        else:
            ziel.write_bytes(inhalt)
    for ziel, quelle in modell_aufgaben():
        if a.check:
            if not ziel.exists() or ziel.read_bytes() != quelle.read_bytes():
                print("ABWEICHUNG:", ziel.name)
                fehler += 1
        else:
            ziel.parent.mkdir(exist_ok=True)
            shutil.copyfile(quelle, ziel)
    print("fertig" if not fehler else f"{fehler} Abweichungen")
    return 1 if fehler else 0


if __name__ == "__main__":
    sys.exit(main())

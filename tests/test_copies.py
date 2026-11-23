"""Kopier-Tests: die kopierten Dateien gleichen ihren Quellen nach dem Ersetzen des Präfixes (übersprungen, wenn die Quelle fehlt, z. B. in der CI), und die Konstantendateien
führen dieselben Namen und Werte wie die Quellen.

Quellen: `zeigernetz-demo` (Präfix znz_ -> nfi_), `zugwahl-lernen-demo` (zwl_ -> nfik_, bei Instanzen, Szenario und Zufallsstrom nfi_), die Vorlage `rng_splitmix64_exact.py` im Website-Repo,
die Modelle aus `rechnen-beim-einsatz-demo` (Zeigernetz A) und `zugwahl-lernen-demo` (Merkmalsnetz mit s). Alle Regeln der Ersetzung stehen in `tools/nfi_kopieren.py`."""

import importlib
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
QUELLE_1 = ROOT.parent / "zeigernetz-demo"
QUELLE_4 = ROOT.parent / "zugwahl-lernen-demo"


def _kopieren():
    spec = importlib.util.spec_from_file_location("nfi_kopieren", ROOT / "tools" / "nfi_kopieren.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


KP = _kopieren()
KOPIEN_1 = [f"nfi_{n}.py" for n in KP.AUS_STUECK_1]
KOPIEN_4 = [f"nfik_{n}.py" for n in KP.AUS_STUECK_4]


def _norm(p: Path) -> bytes:
    return p.read_bytes().replace(b"\r\n", b"\n")


def _quellen_fehlen() -> bool:
    return not (QUELLE_1.exists() and QUELLE_4.exists() and KP.VORLAGE_RNG.exists())


@pytest.mark.parametrize("ziel,inhalt", KP.aufgaben() if not _quellen_fehlen() else [], ids=lambda x: x.name if isinstance(x, Path) else "inhalt")
def test_kopie_gleicht_der_quelle(ziel, inhalt):
    assert _norm(ziel) == inhalt, f"{ziel.name} weicht von der Quelle ab (tools/nfi_kopieren.py --check)"


def test_quellen_vorhanden_oder_uebersprungen():
    if _quellen_fehlen():
        pytest.skip("Quellordner nicht vorhanden (zum Beispiel in der CI): die Kopien sind dort nicht prüfbar")


def test_modelle_gleichen_den_quellen():
    if _quellen_fehlen() or not KP.QUELLE_3.exists():
        pytest.skip("Quellordner nicht vorhanden")
    for ziel, quelle in KP.modell_aufgaben():
        assert ziel.read_bytes() == quelle.read_bytes(), ziel.name


def _modul(pfad: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, pfad)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


def _gebrauchte_namen(dateien) -> set:
    namen = set()
    for d in dateien:
        namen |= set(re.findall(r"(?<![A-Za-z_])C\.([A-Z][A-Z_0-9]*)", (ROOT / d).read_text(encoding="utf-8")))
    return namen


def _konstanten_gleich(eigene: str, quelle_pfad: Path, dateien, ausnahmen=()):
    mine = importlib.import_module(eigene)
    quelle = _modul(quelle_pfad, "quelle_" + eigene)
    fehlt, verschieden = [], []
    for name in sorted(_gebrauchte_namen(dateien) - set(ausnahmen)):
        if not hasattr(mine, name):
            fehlt.append(name)
        elif getattr(mine, name) != getattr(quelle, name):
            verschieden.append(name)
    return fehlt, verschieden


def test_konstanten_aus_stueck_1_gleichen_der_quelle():
    q = QUELLE_1 / "znz_constants.py"
    if not q.exists():
        pytest.skip("zeigernetz-demo nicht vorhanden")
    fehlt, verschieden = _konstanten_gleich("nfi_constants", q, KOPIEN_1)
    assert not fehlt and not verschieden, (fehlt, verschieden)
    gebraucht = _gebrauchte_namen(KOPIEN_1)
    for name in ("C_CLIP", "BATCH", "LERNRATE", "ZWECK_INIT", "ZWECK_TRAIN_X", "ZWECK_TRAIN_S", "MODELL_FORMAT", "SEITE", "FAMILIEN", "MARKE_KLAR", "BOOTSTRAP_ZIEHUNGEN", "OPS_JE_KANDIDAT"):
        assert name in gebraucht


def test_konstanten_aus_stueck_4_gleichen_der_quelle():
    q = QUELLE_4 / "zwl_constants.py"
    if not q.exists():
        pytest.skip("zugwahl-lernen-demo nicht vorhanden")
    fehlt, verschieden = _konstanten_gleich("nfik_constants", q, KOPIEN_4, ausnahmen=("MODELL_ORDNER",))
    assert not fehlt and not verschieden, (fehlt, verschieden)
    gebraucht = _gebrauchte_namen(KOPIEN_4)
    for name in ("H", "K_PARAMS", "Z_PARAMS", "K_MERKMALE", "K_CLIP", "MODELL_FORMAT", "K_LERNRATE"):
        assert name in gebraucht
    import nfik_constants as KC
    assert KC.K_SCHRITTE_FROZEN == _modul(q, "q4").K_SCHRITTE_FROZEN and KC.K_BATCH_TOUREN == 256


def test_zwecke_dieser_demo_liegen_im_reservierten_bereich_und_sind_verschieden():
    import nfi_constants as C
    eigene = [C.ZWECK_AUG, C.ZWECK_MISCH, C.ZWECK_BOOT_ZELLE, C.ZWECK_TEILPROBE, C.ZWECK_WAHL]
    assert all(50 <= z <= 99 for z in eigene) and len(set(eigene)) == len(eigene)
    quellen = [C.ZWECK_INIT, C.ZWECK_TRAIN_X, C.ZWECK_TRAIN_S, C.ZWECK_SAMPLE, C.ZWECK_BOOT, C.ZWECK_ZUFALL, C.ZWECK_INSTANZ, C.ZWECK_START, C.ZWECK_LAUF]
    import nfik_constants as KC
    quellen += [KC.ZWECK_K_INIT, KC.ZWECK_K_INST, KC.ZWECK_K_START, KC.ZWECK_K_ZIEHEN]
    assert not set(eigene) & set(quellen)
    assert len(set(quellen)) == len(quellen)


def test_kein_numpy_random_im_kern():
    """Eigene Dateien importieren weder numpy.random noch random; die Kopien rufen es nicht auf (SplitMix64 überall)."""
    kopien = set(KOPIEN_1 + KOPIEN_4 + ["nfi_rng.py", "modul_frisch.py"])
    for datei in ROOT.glob("*.py"):
        text = datei.read_text(encoding="utf-8")
        assert not re.search(r"(np|numpy)\.random\.|default_rng\(|^\s*import random|^\s*from random import", text, re.M) or datei.name in kopien, datei.name
    for name in kopien:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert not re.search(r"(np|numpy)\.random\.\w+\(|default_rng\(|^\s*import random|^\s*from random import", text, re.M), name

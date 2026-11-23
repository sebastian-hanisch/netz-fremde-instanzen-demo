"""Texte: Wörter der Verbotsliste, ein Block je Display-Gleichung, Prozentzeichen in Formeln, Titel, Zeilen der Erklärtabelle, Annahmenblock in App und README gleich."""

import re
from pathlib import Path

import nfi_constants as C
import nfi_texts as TX

ROOT = Path(__file__).resolve().parent.parent
VERBOTEN = ["trägt", "tragen", "Schwanz", "Saatwert", "TLA", "Parameterwahl", "Verfahrenswahl", "Wahlregeln", "läuft lokal", "Konfiguration", "hub", "sso", "prod", "ass", "qa"]


def _anzeigetexte() -> str:
    teile = [TX.TITEL, TX.UNTERTITEL, TX.INTRO, TX.EINFACHE_WORTE, TX.SO_SIEHT_EIN_NETZ, TX.ERKLAERTABELLE, TX.ZAEHLREGEL, TX.LUPE_TEXT, TX.ASSUMPTIONS_MARKDOWN, TX.FORMEL_LESEHILFE]
    teile += [TX.E0_TITEL, TX.E1_TITEL, TX.E2_TITEL, TX.E3_TITEL, TX.E4_TITEL, TX.E5_TITEL, TX.E1_NUTZT, TX.E2_NUTZT, TX.E3_NUTZT, TX.E4_NUTZT, TX.E5_NUTZT]
    teile += [t for t, _ in TX.FORMELN]
    teile += list(C.PRESET_HELP.values()) + list(C.PRESETS) + list(C.NETZ_ANZEIGE.values())
    return "\n".join(teile)


def test_keine_woerter_der_verbotsliste_in_den_anzeigetexten():
    text = _anzeigetexte()
    for w in VERBOTEN:
        if w in ("hub", "sso", "prod", "ass", "qa"):       # gesperrt sind sie in Datei- und Präfixnamen; im Text nur als ganzes Wort gesucht (nicht in „Matrixprodukte“)
            assert not re.search(r"\b" + w + r"\b", text), w
        else:
            assert w not in text, w


def test_kein_gedankenstrich_im_fliesstext_der_anzeigetexte():
    assert "–" not in _anzeigetexte() and "—" not in _anzeigetexte()


def test_dateinamen_ohne_gesperrte_zeichenfolgen():
    for p in ROOT.iterdir():
        if p.is_file() and not p.name.startswith("."):
            for w in ("hub", "sso", "prod", "qa"):
                assert w not in p.name.lower(), p.name
    assert "ass" not in ROOT.name and "qa" not in ROOT.name


def test_titel_ohne_gesperrte_woerter_und_zufalls_seed_schreibweise():
    assert TX.TITEL.endswith("Das Netz auf anderen Rundtouren")
    for w in ("Konfiguration", "Parameterwahl", "Verfahrenswahl", "Wahlregeln"):
        assert w not in TX.TITEL
    assert "Zufalls-Seed" in TX.INTRO and "Seed" not in TX.INTRO.replace("Zufalls-Seed", "")


def test_jede_gleichung_ein_eigener_block_und_kein_ungeschuetztes_prozentzeichen():
    for _, f in TX.FORMELN:
        assert len(re.findall(r"(?<!\\)%", f)) == 0
        assert "\\\\" not in f or "begin{cases}" in f
        assert "$$" not in f


def test_erklaertabelle_hat_vier_spalten_und_nennt_alle_benannten_verfahren():
    zeilen = [z for z in TX.ERKLAERTABELLE.splitlines() if z.startswith("|")]
    assert all(z.count("|") == 5 for z in zeilen)
    for name in ("Zeigernetz (Pointer Network)", "Merkmalsnetz", "Eingabenormierung", "Nächster Nachbar", "Bootstrap-Intervall"):
        assert any(z.startswith(f"| {name} ") for z in zeilen), name


def test_zahlen_der_konstruktion_in_den_texten_stimmen():
    assert "12 480" in TX.ERKLAERTABELLE and "145 Parameter" in TX.ERKLAERTABELLE and "246 384" in TX.ZAEHLREGEL and "47 242" in TX.ZAEHLREGEL and "47 642" in TX.ZAEHLREGEL
    assert "190 bei n = 20" in TX.ZAEHLREGEL and "19 900 bei n = 200" in TX.ZAEHLREGEL and "960 000" in TX.ZAEHLREGEL and "182 400 000" in TX.ZAEHLREGEL
    assert "162" in TX.ZAEHLREGEL and "31" in TX.ZAEHLREGEL and "1 520" in TX.ZAEHLREGEL


def test_annahmenblock_der_app_steht_in_der_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert TX.ASSUMPTIONS_MARKDOWN in readme


def test_instanzzahlen_im_annahmenblock_gleich_den_konstanten():
    for n in C.GROESSEN:
        assert f"{C.INSTANZEN[n]} bei n = {n}" in TX.ASSUMPTIONS_MARKDOWN

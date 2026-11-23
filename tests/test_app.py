"""End-to-end Rauchtests über Streamlits AppTest: die App lädt mit den Standardeinstellungen, jedes Preset läuft ohne Python-Fehler, die Experimente zeigen Verdikte, das Handbeispiel
rechnet keine Netze. Plotly-Fehler im Browser sieht AppTest nicht (die Browser-Prüfung steht in der README)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import nfi_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))   # unter Last reichen 180 s nicht immer


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at) -> str:
    return " ".join(str(e.value) for kind in (at.success, at.info, at.warning) for e in kind) + " " + " ".join(m.value for m in at.markdown)


def test_app_laedt_ohne_fehler_und_zeigt_die_experimente():
    at = _app()
    ueberschriften = " | ".join(s.value for s in at.subheader)
    for frage in ("Eine Instanz, drei Touren", "kleiner, größer oder verschoben", "bei mehr Orten", "anderer Verteilung", "Wie viele Instanzen", "gemischten Daten", "Wo die Annahmen enden"):
        assert frage in ueberschriften, frage
    assert not at.error


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laeuft_ohne_fehler(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, (name, [str(e) for e in at.exception])
    text = _texte(at)
    if name.startswith("Handbeispiel"):
        assert "Fünf Orte zum Nachrechnen" in text and "Keine Netzläufe" in text
    else:
        assert "Zufalls-Seeds" in text and "Instanzen" in text


def test_verdikte_nennen_k_und_o_oder_die_zaehlregel():
    at = _app()
    v = [str(e.value) for kind in (at.success, at.warning, at.info) for e in kind if "Eine Konstruktion je Instanz" in str(e.value)]
    assert v, "kein Verdikt mit Kostenzeile"
    assert "K = 190 für alle drei gleich" in v[0] and "das 162-Fache" in v[0]

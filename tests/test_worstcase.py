"""Worst-Case am Reglermaximum (Timeout statt Zeit-Assertion, DEMO-PLAYBOOK §1c): der Einzelknopf bei n = 200 mit dem Maximum an Instanzen und fünf Seeds, E1 bei n = 50 mit 30 Instanzen."""

import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

import nfi_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def test_einzelknopf_n_200_am_maximum_unter_dem_zeitlimit():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.sidebar.select_slider(key="n_select").set_value(200)
    at.run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]
    at.slider(key="e2_m").set_value(C.E2_KNOPF_M_MAX)
    at.run(timeout=120)
    at.button(key="e2_live").click().run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("mittlere Lücke" in m.value for m in at.markdown)


def test_e1_n_50_mit_30_instanzen_am_maximum():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.sidebar.select_slider(key="n_select").set_value(50)
    at.run(timeout=120)
    at.slider(key="e1_m").set_value(C.LIVE_M_MAX)
    at.run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]

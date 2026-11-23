"""Gemeinsame Fixtures: die Fünf-Orte-Instanz des Handbeispiels (von Hand gerechnet, `nfi_hand.py`), an der jede Einheit geprüft wird (DEMO-PLAYBOOK §1b)."""

import os
import sys
from pathlib import Path

import pytest

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

HAND = np.array([[0.0, 0.0], [0.4, 0.2], [1.0, 0.2], [0.8, 0.8], [0.2, 0.6]])   # P0 bis P4 des Plans, hartcodiert
HAND_NN_TOUR = [0, 1, 4, 3, 2]
HAND_OPT_LAENGE = 3 * 0.4 ** 0.5 + 0.6 + 0.2 ** 0.5                                # 2,9446 von Hand: drei Kanten sqrt(0,4), eine 0,6, eine sqrt(0,2)


@pytest.fixture
def hand():
    return HAND.copy()


def zufalls_D(n: int, count: int, seed: int = 0, familie: str = "gleich"):
    """(X, D) für `count` Instanzen der Quelle (Zufalls-Seed seed bis seed + count - 1)."""
    import nfi_transform as TR
    X = TR.testmenge(familie, n, count, ab=seed)
    return X, TR.abstaende(X)

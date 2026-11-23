"""Handbeispiel: fünf Orte im Einheitsquadrat zum Nachrechnen (Plan, Abschnitt „Handbeispiel von Hand“), in Schleifen und ohne die Vektorfassungen der App.

P0 (0; 0), P1 (0,4; 0,2), P2 (1; 0,2), P3 (0,8; 0,8), P4 (0,2; 0,6). Enthält: Nächster Nachbar mit Zählung der Kandidaten (K = 10), das Optimum durch Aufzählung aller 24 Touren
ab P0 (12 verschiedene Rundtouren), die Spielzeug-Zeigerstufe (nur die Form u = 10 tanh(<x_i, x_j> / sqrt(d)), kein trainiertes Netz) und die sieben Merkmale des Merkmalsnetzes von Hand
(mit der Skala s der Instanz und mit der festen Skala s0). Die Tests (`tests/test_hand.py`) halten diese Rechnung gegen die Vektorfassungen und die Werte des Plans."""

import itertools
import math

import numpy as np

import nfi_constants as C

PUNKTE = np.array([[0.0, 0.0], [0.4, 0.2], [1.0, 0.2], [0.8, 0.8], [0.2, 0.6]])


def abstaende(X: np.ndarray) -> np.ndarray:
    """Abstandsmatrix aus Koordinaten (Schleifenfassung)."""
    n = len(X)
    return np.array([[math.hypot(X[i, 0] - X[j, 0], X[i, 1] - X[j, 1]) for j in range(n)] for i in range(n)])


def tour_laenge(D: np.ndarray, t) -> float:
    return float(sum(D[t[i], t[(i + 1) % len(t)]] for i in range(len(t))))


def nn_tour_gezaehlt(D: np.ndarray, start: int = 0):
    """Nächster Nachbar mit Zählung der Kandidatenbewertungen; Gleichstand: kleinster Index. Gibt (Tour, K)."""
    frei = [j for j in range(len(D)) if j != start]
    t, k = [start], 0
    while frei:
        cur = t[-1]
        best = None
        for j in frei:
            k += 1
            if best is None or D[cur, j] < D[cur, best] - 1e-12:
                best = j
        t.append(best)
        frei.remove(best)
    return t, k


def optimum_aufzaehlung(D: np.ndarray):
    """Kürzeste Tour durch Aufzählung aller (n−1)! Touren ab Ort 0: (Länge, Tour, Zahl der Touren, Zahl der verschiedenen Rundtouren)."""
    n = len(D)
    alle = [(tour_laenge(D, [0] + list(p)), [0] + list(p)) for p in itertools.permutations(range(1, n))]
    best = min(alle, key=lambda a: a[0])
    return best[0], best[1], len(alle), len({round(a[0], 12) for a in alle})


def zeiger_logits_spielzeug(X: np.ndarray, cur: int, frei: list, c: float = C.C_CLIP) -> np.ndarray:
    """Spielzeug-Zeigerstufe: u_j = C tanh(<x_cur, x_j> / sqrt(d)), d = 2 (nur die Form „Beschneidung mal tanh“, kein trainiertes Netz)."""
    d = X.shape[1]
    return np.array([c * math.tanh(float(X[cur] @ X[j]) / math.sqrt(d)) for j in frei])


def spannweite(u: np.ndarray) -> float:
    return float(u.max() - u.min())


def skala_s(D: np.ndarray) -> float:
    """Skala s: Mittel der Abstände zum nächsten Nachbarn."""
    M = D.copy()
    np.fill_diagonal(M, np.inf)
    return float(M.min(1).mean())


def merkmale(D: np.ndarray, s: float, cur: int, start: int, frei: list, n_besucht: int) -> dict:
    """Die sieben Merkmale x1 bis x7 je freiem Ort j (Schleifen)."""
    n = len(D)
    dmin = min(D[cur, j] for j in frei)
    geordnet = sorted(frei, key=lambda j: D[cur, j])
    out = {}
    for j in frei:
        x1 = min(D[cur, j] / s, 6.0)
        x2 = min(geordnet.index(j), 8) / 8.0
        x3 = min(D[start, j] / s, 6.0)
        x4 = min(min(D[j, k] for k in frei if k != j) / s, 6.0)
        x5 = min(sum(1 for k in frei if k != j and D[j, k] <= 2 * s), 8) / 8.0
        x6 = n_besucht / n
        x7 = min(D[cur, j] / dmin - 1.0, 6.0)
        out[j] = np.array([x1, x2, x3, x4, x5, x6, x7])
    return out

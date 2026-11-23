"""Instanzen und Wechsel: Skala und Lage ändern, Testmengen aller Familien, Skala s, Nächster Nachbar und Tourlängen für viele Instanzen auf einmal (numpy).

Wechsel (`wechsel`): gleichmäßiges Skalieren und Verschieben der Koordinaten, x -> a x + b (a > 0). Der Nächste Nachbar und jede Tour behalten ihre Reihenfolge, die Tourlänge
skaliert mit a; die Netze sehen die veränderten Koordinaten (oder deren Normierung, oder die Abstände geteilt durch s). Bewertet wird immer auf der **Basisinstanz**."""

import numpy as np

import nfi_constants as C
import nfi_instances as I
import nfi_scenario as SC


def wechsel(X: np.ndarray, art: str) -> np.ndarray:
    """Koordinaten (..., n, 2) unter dem Wechsel `art` (basis, klein, gross, versch)."""
    X = np.asarray(X, dtype=np.float64)
    if art == "basis":
        return X
    if art == "klein":
        return C.FAKTOR_KLEIN * X
    if art == "gross":
        return C.FAKTOR_GROSS * X
    if art == "versch":
        return X + np.asarray(C.VERSCHIEBUNG)
    raise ValueError(f"unbekannter Wechsel {art!r}")


def wechsel_frei(X: np.ndarray, faktor: float, vx: float, vy: float) -> np.ndarray:
    """Beliebiger Wechsel x -> faktor x + (vx, vy) (Regler der Seitenleiste)."""
    return faktor * np.asarray(X, dtype=np.float64) + np.array([vx, vy], dtype=np.float64)


def skala_s(D: np.ndarray) -> np.ndarray:
    """Skala s: mittlerer Abstand zum nächsten Nachbarn; D (R, n, n) -> (R,) oder (n, n) -> Zahl."""
    n = D.shape[-1]
    return (D + np.diag(np.full(n, np.inf))).min(-1).mean(-1)


def testmenge(familie: str, n: int, count: int, ab: int = 0) -> np.ndarray:
    """Testinstanzen (count, n, 2) im Einheitsquadrat: `make_instance(familie, n, seed)` für Zufalls-Seed `ab` bis `ab + count − 1`, durch 1000 geteilt."""
    return np.stack([I.instance_unit(familie, n, s) for s in range(ab, ab + count)])


def abstaende(X: np.ndarray) -> np.ndarray:
    """Abstandsmatrizen (R, n, n) aus Koordinaten (R, n, 2)."""
    return I.distance_matrix_np(np.asarray(X, dtype=np.float64))


def nn_touren(D: np.ndarray, start: int = 0) -> np.ndarray:
    """Nächster Nachbar für R Instanzen D (R, n, n) ab `start`: Touren (R, n). Gleichstand: kleinerer Index (wie `nfi_gegner.nn_tour_k`)."""
    R, n, _ = D.shape
    ar = np.arange(R)
    unv = np.ones((R, n), dtype=bool)
    unv[:, start] = False
    cur = np.full(R, start, dtype=np.int64)
    tour = [cur.copy()]
    for _ in range(n - 1):
        a = np.where(unv, D[ar, cur, :], np.inf).argmin(1)
        unv[ar, a] = False
        cur = a
        tour.append(a.copy())
    return np.stack(tour, axis=1)


def tour_laengen(D: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Geschlossene Tourlängen: D (R, n, n), Touren (R, n) -> (R,)."""
    ar = np.arange(D.shape[0])[:, None]
    return D[ar, T, np.roll(T, -1, axis=1)].sum(1)


def ist_permutation(T: np.ndarray) -> bool:
    """Ist jede Tour (letzte Achse) eine Permutation von 0..n−1?"""
    return bool(np.all(np.sort(T, axis=-1) == np.arange(T.shape[-1])))


def zufalls_instanz(familie: str, n: int, seed: int) -> np.ndarray:
    """Eine Instanz (n, 2) im Einheitsquadrat (Zufalls-Seed der App, 0 bis 99)."""
    return I.instance_unit(familie, n, seed)


def hand_instanz() -> np.ndarray:
    """Das Handbeispiel: fünf Orte im Einheitsquadrat (P0 bis P4 des Plans)."""
    return np.array([[0.0, 0.0], [0.4, 0.2], [1.0, 0.2], [0.8, 0.8], [0.2, 0.6]])


def tour_cost_liste(D, t) -> float:
    """Tourlänge für eine Abstandsliste (Quelle `nfi_scenario.tour_cost`)."""
    return SC.tour_cost(D, t)

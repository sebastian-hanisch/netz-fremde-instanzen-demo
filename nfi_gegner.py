"""Klassische Vergleichsverfahren in einfachster Fassung, reines Python auf Listen, mit Zähler der Kandidatenbewertungen (K).

* `nn_tour_k`: Nächster Nachbar ab `start` (Gleichstand: kleinerer Index); jede Prüfung eines freien Ortes ist eine Kandidatenbewertung: n(n-1)/2 je Tour.
* `two_opt_k`: 2-opt mit erster Verbesserung (ohne Nachbarlisten, ohne Or-opt); jedes bewertete Paar von Kanten ist eine Kandidatenbewertung,
  eine volle Prüfrunde hat n(n-3)/2 (Paare i < j ohne benachbarte Kanten). Läuft bis zum 2-opt-Optimum.
* `zufallstour`: Zufallsanordnung der Orte 1..n-1 hinter Ort 0.

Or-opt, Neustarts und Ketten bauen die Folge-Stücke selbst (Zählweise der verschiedenen Züge: nfi_cost.m_verschiedene_zuege)."""

from nfi_scenario import tour_cost


def nn_tour_k(D, start: int = 0):
    """(Tour, Zahl der Kandidatenbewertungen) des Nächsten Nachbarn ab `start`."""
    n = len(D)
    seen = [False] * n
    tour = [start]
    seen[start] = True
    cur = start
    k = 0
    for _ in range(n - 1):
        row = D[cur]
        best, bd = -1, 1e300
        for j in range(n):
            if not seen[j]:
                k += 1
                if row[j] < bd:
                    bd, best = row[j], j
        tour.append(best)
        seen[best] = True
        cur = best
    return tour, k


def two_opt_k(D, tour):
    """(Tour, Länge, Zahl der Kandidatenbewertungen): 2-opt mit erster Verbesserung bis zum Optimum; die Eingabe wird nicht verändert."""
    tour = list(tour)
    n = len(tour)
    cur = tour_cost(D, tour)
    k = 0
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            a = tour[i]
            b = tour[i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c = tour[j]
                dd = tour[(j + 1) % n]
                delta = D[a][c] + D[b][dd] - D[a][b] - D[c][dd]
                k += 1
                if delta < -1e-12:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    cur += delta
                    improved = True
                    a = tour[i]
                    b = tour[i + 1]
    return tour, tour_cost(D, tour), k


def zufallstour(rng, n: int) -> list[int]:
    """Zufällige Rundtour: Ort 0, dahinter eine Zufallsanordnung der übrigen Orte (`rng` ist ein SplitMix64)."""
    rest = list(range(1, n))
    rng.shuffle(rest)
    return [0] + rest

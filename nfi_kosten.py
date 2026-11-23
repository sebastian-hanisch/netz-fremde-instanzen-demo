"""Kosten je Konstruktion nach der Zählregel der Linie (Zuschnitt 2 Abschnitt 10, Q5): Kandidatenbewertungen K und grobe Operationen O nebeneinander.

K: eine Konstruktion kostet n (n - 1) / 2 Kandidatenbewertungen, für den Nächsten Nachbarn, das Zeigernetz und das Merkmalsnetz gleich (`nfi_cost.k_konstruktion`). K begünstigt die Netze
(ein Zeigerschritt oder eine Merkmalszeile kostet mehr als ein Vergleich).
O: Multiplikation-Addition = 1 Operation; Zeigernetz aus den Formen der Matrixprodukte einzeln gezählt (`nfi_cost.ops_netz_tour`, n = 20, d = 32: 246 384), Nächster Nachbar 8 je
Kandidat, Merkmalsnetz K: n - 1 Merkmalsbildungen zu 2 n^2 + n (8 + log2 n) und n (n - 1) / 2 Zeilen zu 144 (Quelle Stück 4: `O_JE_ZEILE_K = 144`, Stück 3: mit s zusätzlich n^2 für die
Skala). O benachteiligt die Netze (die Annahme „8 je Kandidat“ ist ein Zehnerpotenz-Wert). Das Training steht als dritter Posten daneben und geht nie in K oder O."""

import math

import nfi_constants as C
import nfi_cost as K0
import nfik_constants as KC

O_JE_ZEILE_MERKMAL = KC.K_MERKMALE * KC.H + KC.H + KC.H   # 144: Netz 7-16-1 je Zeile (Merkmalsbildung nicht mitgezählt)


def k_konstruktion(n: int) -> int:
    """Kandidatenbewertungen einer Konstruktion: n (n - 1) / 2."""
    return K0.k_konstruktion(n)


def ops_nn(n: int) -> int:
    """O des Nächsten Nachbarn: 8 je Kandidatenbewertung."""
    return int(K0.ops_kandidaten(k_konstruktion(n)))


def ops_zeiger(n: int, d: int = C.D_ZEIGER) -> int:
    """O einer Greedy-Tour des Zeigernetzes (Kodierer plus Dekodierer, Formen einzeln gezählt)."""
    return K0.ops_netz_tour(n, d)


def merkmalsbildung_schritt(n: int) -> float:
    """O der Merkmalsbildung je Schritt: 2 n^2 + n (8 + log2 n)."""
    return 2.0 * n * n + n * (8 + math.log2(n))


def ops_merkmal(n: int, mit_s: bool = False) -> int:
    """O einer Greedy-Tour des Merkmalsnetzes: (n - 1) Merkmalsbildungen plus n (n - 1) / 2 Zeilen zu 144; mit der Skala s der Instanz zusätzlich n^2 (Stück 3 zählt so: 47 642 bei n = 20),
    ohne s (feste Skala) 47 242."""
    o = (n - 1) * merkmalsbildung_schritt(n) + k_konstruktion(n) * O_JE_ZEILE_MERKMAL + (n * n if mit_s else 0)
    return int(o)


def kosten_je_konstruktion(n: int) -> dict:
    """K und O je Konstruktion bei n Orten: {"K", "O_nn", "O_zeiger", "O_merkmal", "O_merkmal_mit_s", "f_zeiger", "f_merkmal"} (Faktoren gegen den Nächsten Nachbarn, Merkmalsnetz ohne s)."""
    o_nn, o_z, o_m = ops_nn(n), ops_zeiger(n), ops_merkmal(n)
    return {"K": k_konstruktion(n), "O_nn": o_nn, "O_zeiger": o_z, "O_merkmal": o_m, "O_merkmal_mit_s": ops_merkmal(n, True), "f_zeiger": o_z / o_nn, "f_merkmal": o_m / o_nn}


def training_posten() -> dict:
    """Training als dritter Posten: gesampelte Touren und Merkmalszeilen (nie in K oder O)."""
    return {"zeiger_touren": C.TOUREN_ZEIGER, "merkmal_touren": C.TOUREN_MERKMAL, "merkmal_zeilen": C.ZEILEN_MERKMAL}


def live_o(n: int, netze: tuple, seeds: int, m: int) -> float:
    """Grobe Operationen einer Live-Rechnung: m Instanzen x Summe der O je Netz x Zahl der Seeds."""
    o = 0
    for netz in netze:
        o += ops_zeiger(n) if netz in C.ZEIGER_NETZE else ops_merkmal(n, netz == "Bmit")
    return float(m) * o * seeds


def cap_live(n: int, netze: tuple, seeds: int, m: int, o_max: float = C.LIVE_O_MAX, m_max: int = C.LIVE_M_MAX) -> int:
    """Größtes m <= m_max mit live_o <= o_max (mindestens 1)."""
    m = max(1, min(int(m), int(m_max)))
    while m > 1 and live_o(n, netze, seeds, m) > o_max:
        m -= 1
    return m

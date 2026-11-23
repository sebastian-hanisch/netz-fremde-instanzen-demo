"""Befunde aus der eingefrorenen Messung: alle Kennzahlen, die die README und die Tests nennen, an einer Stelle gerechnet (`befunde(fr)`), mit den Zählungen über die Zellen.

Die README-Tabelle „Zahl → Test“ und `tests/test_claims.py` lesen dieselben Größen; `tools/nfi_zahlen.py` druckt sie für die README."""

import nfi_constants as C
import nfi_evaluation as V
import nfi_experiments as EX


def _bruch(klassen: dict) -> tuple:
    k = V.zaehle_klassen(klassen.values())
    return k["bricht"], k["schlaegt"], k["gesamt"]


def befunde(fr: EX.Frozen) -> dict:
    """Wörterbuch aller Kennzahlen (Lücken in Prozent, Unterschiede in Prozentpunkten, Zählungen als (bricht, schlägt, gesamt))."""
    b = {}
    b["nn_luecke_20"] = fr.luecke_nn("gleich", 20)
    for netz in fr.netze():
        b[f"luecke_{netz}_20"] = fr.zelle("gleich", 20, netz)["luecke"]
        b[f"basis_{netz}"] = _bruch(fr.basis_klassen(netz))
        b[f"gross_{netz}"] = _bruch(fr.basis_klassen(netz, 100))
        if netz in ("A", "Ant", "An", "Bmit", "Bohne", "Aaug", "Amix"):
            b[f"wechsel_{netz}"] = _bruch(fr.wechsel_klassen(netz))
    for netz in ("A", "Bohne", "Aaug"):
        if fr.hat_netz(netz):
            b[f"wechsel_luecke_ueber_basis_{netz}"] = fr.wechsel_luecke_gleich_basis(netz)
    for n in (20, 50):
        z = fr.zelle("gleich", n, "A", "gross")
        b[f"A_gross_luecke_{n}"] = z["luecke"]
        b[f"A_gross_unterschied_{n}"] = z["v"]
        b[f"nn_luecke_{n}"] = fr.luecke_nn("gleich", n)
    for n in (100, 200):
        b[f"Bmit_unterschied_{n}"] = fr.zelle("gleich", n, "Bmit")["v"]
    for fam in ("gleich", "cluster"):
        ks = [n for n in C.GROESSEN if fr.zelle(fam, n, "A")["klasse"] == "bricht"]
        b[f"A_erstes_n_{fam}"] = ks[0] if ks else None
    abst = []
    for n in C.GROESSEN:
        for fam in C.FAMILIEN:
            base = fr.zelle(fam, n, "A")["luecke"]
            for netz in ("Ant", "An"):
                abst.append(abs(fr.zelle(fam, n, netz)["luecke"] - base))
    b["norm_kosten_max"] = max(abst)
    b["s_verhaeltnis"] = {n: fr.d["diagnose"]["s_verhaeltnis"][f"gleich_{n}"] for n in C.GROESSEN}
    dg = fr.d["diagnose"]
    b["saettigung"] = {art: dg["saettigung"][art]["mittel"] for art in C.WECHSEL}
    b["kappung_x1"] = {art: dg["kappung_bohne"][art]["x1"] for art in C.WECHSEL}
    b["invarianz_min"] = min([v for w in fr.d["wechsel"].values() for v in w.get("invarianz", {}).values()] or [None])
    b["unterboten"] = {k: v["unterboten_anteil"] for k, v in fr.d["base"].items() if C.INSTANZEN and int(k.split("_")[1]) > 20}
    return b

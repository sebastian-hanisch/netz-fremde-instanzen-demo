"""Das Netz auf anderen Rundtouren: Skala, Lage, Größe und Verteilung - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zwei von Hand in numpy gebaute Netze (Zeigernetz, Merkmalsnetz), beide bei 20 Orten gleichverteilt trainiert und vorgerechnet, bauen je Instanz eine greedy-Tour und treten gegen den Nächsten
Nachbarn an, wenn die Instanz anders aussieht als im Training: kleiner oder größer gezeichnet, verschoben (Skala und Lage, mit einer Normierung der Eingabe als Schalter), mit mehr Orten
oder anders verteilt (Größe und Verteilung). Aufbau nach DEMO-PLAYBOOK §2: Kopf, Einleitung, Erklär-Expander, Primäransicht (eine Instanz), fünf Experimente mit Verdikt, Kosten, Grenzen, Formeln.
In der App wird nicht trainiert.

Lauffähig mit: streamlit run app.py
"""
# ruff: noqa: E402  (die BLAS-Umgebungsvariablen müssen vor dem ersten Import von numpy stehen)

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # vor dem ersten Import von numpy: ein BLAS-Thread

import numpy as np
import streamlit as st

import modul_frisch
import nfi_constants as C
import nfi_diagnose as DG
import nfi_eval as EV
import nfi_evaluation as V
import nfi_exact as EXA
import nfi_experiments as EX
import nfi_hand as HA
import nfi_kosten as KO
import nfi_layout as L
import nfi_models as MO
import nfi_net as N
import nfi_presets as PR
import nfi_texts as TX
import nfi_transform as TR
import nfi_visualization as VIS

modul_frisch.aktualisieren(__file__)

st.set_page_config(page_title="Das Netz auf anderen Rundtouren – Sebastian Hanisch", layout="wide")

ZEIGER_VON_NORM = {"aus": "A", "einsatz": "Ant", "training": "An"}
MERKMAL_VON_SKALA = {"mit": "Bmit", "ohne": "Bohne"}


# ---------------------------------------------------------------------------------------------------------------- Zwischenspeicher
@st.cache_resource(show_spinner=False)
def _frozen():
    return EX.Frozen.laden()


@st.cache_data(show_spinner=False)
def _held_karp_tour(fam, n, seed):
    X = TR.zufalls_instanz(fam, n, seed)
    return EXA.held_karp_tour(TR.abstaende(X[None])[0])


@st.cache_data(show_spinner=False)
def _e1_live(fam, n, m, seeds, zeiger, merkmal, wechsel):
    """{(Wechsel, Netz): Zelle}; die gegen Skala und Lage invarianten Netze werden nur auf der Basisinstanz gerechnet."""
    fr = _frozen()
    out = {}
    for w in wechsel:
        for netz in (zeiger, merkmal):
            if netz in C.INVARIANTE_NETZE and w != "basis" and ("basis", netz) in out:
                out[(w, netz)] = out[("basis", netz)]
                continue
            out[(w, netz)] = EX.live_zelle(fam, n, netz, list(range(m)), seeds, w if isinstance(w, str) else tuple(w), fr)
    return out


@st.cache_data(show_spinner=False)
def _e2_live(fam, n, m, seeds, zeiger, merkmal):
    fr = _frozen()
    return {netz: EX.live_zelle(fam, n, netz, list(range(m)), seeds, "basis", fr) for netz in (zeiger, merkmal)}


@st.cache_data(show_spinner=False)
def _diagnose_live(fam, n, seeds_inst, faktor, vx, vy, seeds_netz):
    X = np.stack([TR.zufalls_instanz(fam, n, s) for s in seeds_inst])
    Xw = TR.wechsel_frei(X, faktor, vx, vy)
    sat = float(np.mean([DG.saettigung_anteil(P, Xw) for P in MO.lade_alle("A", range(seeds_netz))]))
    D = TR.abstaende(Xw)
    kap = [DG.kappung_anteil(p, D, np.full(len(D), C.S0)) for p in MO.lade_alle("Bohne", range(seeds_netz))]
    return sat, {k: float(np.mean([x[k] for x in kap])) for k in kap[0]}


def zeige(art_text):
    art, text = art_text
    getattr(st, art)(text)


def pct(x, stellen=1):
    return V.de(x, stellen) + " %"


# ---------------------------------------------------------------------------------------------------------------- Kopf, Einleitung, Erklärung
st.html(L.kopf_css(L.dunkles_design()))
with st.container(key="kopf"):
    st.title(TX.TITEL)
    st.caption(TX.UNTERTITEL)
st.markdown(TX.INTRO)

with st.expander("💬 In einfachen Worten"):
    st.markdown(TX.EINFACHE_WORTE)

with st.expander("So sieht ein Netz eine fremde Instanz", expanded=True):
    st.markdown(TX.SO_SIEHT_EIN_NETZ)

with st.expander("Wie hier gezählt wird (K und O)"):
    st.markdown(TX.ZAEHLREGEL)

with st.expander("Benannte Verfahren kurz erklärt"):
    st.markdown(TX.ERKLAERTABELLE)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Auswahl wider, einfach kopieren, um ein Szenario zu teilen.")

PR.load_permalink_settings()
PR.init_session_state_defaults()
fr = _frozen()

# ---------------------------------------------------------------------------------------------------------------- Seitenleiste
with st.sidebar:
    PR.sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Instanz")
    familie = st.selectbox("Familie", options=PR.options("familie_select"), format_func=C.FAMILIE_ANZEIGE.get, key="familie_select",
                           help="Gleichverteilt: Orte zufällig im Quadrat. Geclustert: fünf Haufen (Gauß, σ = 4 % der Seite). Gitter: Gitterpunkte mit Rauschen. Die Netze sind nur auf gleichverteilten Orten trainiert.")
    n = st.select_slider("Städtezahl n", options=PR.options("n_select"), key="n_select",
                         help="Die Netze sind bei n = 20 trainiert. n = 5 ist nur das Handbeispiel (keine Netzläufe). Die Live-Experimente rechnen höchstens bis n = 50.")
    hand = n == C.N_HAND
    seed = st.number_input("Zufalls-Seed der Instanz", *PR.bounds("seed_input"), key="seed_input", step=1, disabled=hand,
                           help="Zufalls-Seed 0 bis 99: zugleich der Index der eingefrorenen Testinstanz (für sie liegen Referenz und Messung vor, soweit n und Instanzzahl es zulassen).")
    st.button("🎲 Neue Instanz", width="stretch", on_click=PR.randomize_instance, disabled=hand)
    faktor = st.select_slider("Skalenfaktor", options=PR.options("faktor_select"), key="faktor_select", format_func=lambda x: V.de(x, 2),
                              help="Alle Koordinaten werden mit diesem Faktor multipliziert (Standard 1: Einheitsquadrat). Die Tour des Nächsten Nachbarn bleibt dieselbe.")
    vx = st.slider("Verschiebung x", *PR.bounds("vx_slider"), key="vx_slider", help="Alle Koordinaten werden um diesen Wert in x-Richtung verschoben.")
    vy = st.slider("Verschiebung y", *PR.bounds("vy_slider"), key="vy_slider", help="Alle Koordinaten werden um diesen Wert in y-Richtung verschoben.")
    st.header("⚙️ Netze")
    norm = st.radio("Eingabenormierung des Zeigernetzes", options=PR.options("norm_select"), format_func=C.NORM_ANZEIGE.get, key="norm_select", disabled=hand,
                    help="aus: rohe Koordinaten (Netz A). Beim Einsatz: dieselben Gewichte, die Eingabe wird auf das Einheitsquadrat normiert (A+nt). Auch im Training: ein bei normierter Eingabe trainiertes Netz (An).")
    skala = st.radio("Skala des Merkmalsnetzes", options=PR.options("skala_select"), format_func=C.SKALA_ANZEIGE.get, key="skala_select", disabled=hand,
                     help="Skala s der Instanz: Abstände werden durch den mittleren Abstand zum nächsten Nachbarn geteilt (B mit s). Feste Skala s0 = 0,1239: das Netz B ohne s, trainiert mit der festen Skala.")
    netz_seed = st.number_input("Zufalls-Seed des Netzes", *PR.bounds("netz_seed_input"), key="netz_seed_input", step=1, disabled=hand,
                                help="Welches der fünf vorgerechneten Modelle (0 bis 4) die Karten der Primäransicht zeigen (Gewichts- und Daten-Zufalls-Seed gepaart).")
    seeds_zahl = st.select_slider("Zahl der Zufalls-Seeds (Live-Läufe)", options=PR.options("seeds_zahl_select"), key="seeds_zahl_select", disabled=hand,
                                  help="Die Live-Läufe in den Experimenten mitteln über die ersten k der fünf Modelle; die eingefrorene Messung nimmt immer alle fünf.")
PR.sync_query_params({k: st.session_state[k] for k in PR.SETTING_SPECS})

zeiger = ZEIGER_VON_NORM[norm]
merkmal = MERKMAL_VON_SKALA[skala]
wechsel_aktiv = (faktor, vx, vy)
wechsel_ist_basis = abs(faktor - 1.0) < 1e-12 and vx == 0 and vy == 0
st.caption(f"Netze: {C.NETZ_ANZEIGE[zeiger]} und {C.NETZ_ANZEIGE[merkmal]}. Wechsel der Seitenleiste: Faktor {V.de(faktor, 2)}, Verschiebung ({vx}; {vy}).")
if fr is None:
    st.error("Die eingefrorene Messung (nfi_frozen.json) fehlt: Experimente und Tabellen sind nicht verfügbar.")
if not (MO.vorhanden(zeiger) and MO.vorhanden(merkmal)) and not hand:
    st.warning("Für die gewählte Variante liegen nicht alle vorgerechneten Modelle vor.")

# ---------------------------------------------------------------------------------------------------------------- Primäransicht E0
st.subheader(TX.E0_TITEL)
if hand:
    X = HA.PUNKTE
    D = HA.abstaende(X)
    t_nn, k_nn = HA.nn_tour_gezaehlt(D)
    L_opt, t_opt, _, _ = HA.optimum_aufzaehlung(D)
    L_nn = HA.tour_laenge(D, t_nn)
    c1, c2 = st.columns(2)
    c1.plotly_chart(VIS.fig_karte(X, t_nn, f"Nächster Nachbar: Länge {V.de(L_nn, 4)}", C.COLOR_NN), key="e0_hand_nn", width="stretch")
    c2.plotly_chart(VIS.fig_karte(X, t_opt, f"Exakt (Aufzählung aller 24 Touren): Länge {V.de(L_opt, 4)}", C.COLOR_OPT), key="e0_hand_opt", width="stretch")
    st.markdown(f"Fünf Orte zum Nachrechnen. **Nächster Nachbar** {t_nn}, Länge {V.de(L_nn, 4)}, K = {k_nn} Kandidatenbewertungen (= 5·4/2); **exakt** {t_opt}, Länge {V.de(L_opt, 4)}; "
                f"Lücke des Nächsten Nachbarn {pct(100 * (L_nn - L_opt) / L_opt)}. Keine Netzläufe: die Netze sind bei n = 20 trainiert und für n ≥ 10 gemessen. Die Lupe in Experiment 1 rechnet die Merkmale und die Sättigung an diesen fünf Orten von Hand.")
else:
    X = TR.zufalls_instanz(familie, n, seed)
    Xw = TR.wechsel_frei(X, faktor, vx, vy)
    D = TR.abstaende(X[None])
    t_nn = TR.nn_touren(D)[0]
    L_nn = float(TR.tour_laengen(D, t_nn[None])[0])
    if n <= 20:
        L_ref, _t_ref = _held_karp_tour(familie, n, seed)
        ref_art = "exakt"
    elif fr is not None and seed < fr.m(n):
        L_ref, ref_art = float(fr.ref(familie, n)[seed]), "beste bekannte Lösung"
    else:
        L_ref, ref_art = None, "keine Referenz"
    modell_z = MO.lade(zeiger, netz_seed)[0]
    modell_m = MO.lade(merkmal, netz_seed)[0]
    t_z = EV.touren_zeiger(zeiger, modell_z, Xw[None])[0]
    t_m = EV.touren_merkmal(merkmal, modell_m, Xw[None])[0][0]
    lens = {"Nächster Nachbar": L_nn, C.NETZ_KURZ[zeiger]: float(TR.tour_laengen(D, t_z[None])[0]), C.NETZ_KURZ[merkmal]: float(TR.tour_laengen(D, t_m[None])[0])}
    st.caption(f"Eingabe der Netze (nach dem Wechsel, vor der Normierung): x von {V.de(Xw[:, 0].min(), 2)} bis {V.de(Xw[:, 0].max(), 2)}, y von {V.de(Xw[:, 1].min(), 2)} bis {V.de(Xw[:, 1].max(), 2)}. "
               "Gezeichnet und bewertet wird alles auf der Basisinstanz.")
    schritt = st.slider("Schritt t (Karten zeigen die Tour bis zum Schritt)", 0, n, n, key=f"e0_schritt_{n}", help="n: ganze Tour (mit Rückkehr zum Start).")
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(VIS.fig_karte(X, t_nn, "Nächster Nachbar", C.COLOR_NN, schritt), key="e0_nn", width="stretch")
    c2.plotly_chart(VIS.fig_karte(X, t_z, C.NETZ_KURZ[zeiger], C.NETZ_FARBE[zeiger], schritt), key="e0_z", width="stretch")
    c3.plotly_chart(VIS.fig_karte(X, t_m, C.NETZ_KURZ[merkmal], C.NETZ_FARBE[merkmal], schritt), key="e0_m", width="stretch")
    kk = KO.kosten_je_konstruktion(n)
    zeilen = ["| Verfahren | Länge | Lücke zur Referenz | Unterschied zum Nächsten Nachbarn | K | O |", "|---|---|---|---|---|---|"]
    o_je = {"Nächster Nachbar": kk["O_nn"], C.NETZ_KURZ[zeiger]: kk["O_zeiger"], C.NETZ_KURZ[merkmal]: KO.ops_merkmal(n, merkmal == "Bmit")}
    for name, ll in lens.items():
        luecke = pct(100 * (ll - L_ref) / L_ref) if L_ref else "–"
        diff = ("0" if name == "Nächster Nachbar" else (V.de_vz(100 * (ll - L_nn) / (L_ref or L_nn), 1) + (" Prozentpunkte" if L_ref else " % der Länge des Nächsten Nachbarn")))
        zeilen.append(f"| {name} | {V.de(ll, 3)} | {luecke} | {diff} | {V.zahl(kk['K'])} | {V.zahl(o_je[name])} |")
    st.markdown("\n".join(zeilen))
    st.caption(f"Referenz: {ref_art}" + (f" ({V.de(L_ref, 3)})" if L_ref else "") + f". Netze: Zufalls-Seed des Netzes {netz_seed}. Alles Weitere steht in den Experimenten.")
    for netz in (zeiger, merkmal):
        d = 100 * (lens[C.NETZ_KURZ[netz]] - L_nn) / (L_ref or L_nn)
        zeige(V.verdict_instanz(V.NETZ_NAME["zeiger" if netz in C.ZEIGER_NETZE else "merkmal"], d, "Prozentpunkte" if L_ref else "Prozent der Länge des Nächsten Nachbarn"))

# ---------------------------------------------------------------------------------------------------------------- E1 Skala und Lage
st.subheader(TX.E1_TITEL)
st.caption(TX.E1_NUTZT)
kosten_n = KO.kosten_je_konstruktion(min(max(n, 10), 200) if not hand else 20)
if fr is not None:
    e1a, e1b, e1c = st.columns(3)
    m_wunsch = e1a.slider("Zahl der Instanzen m", 3, C.LIVE_M_MAX, C.DEFAULT_E1_M, key="e1_m", help="Live gerechnet an den Instanzen mit Zufalls-Seed 0 bis m − 1; die Rechengrenze kann m kürzen.")
    alle_w = e1b.checkbox("alle drei Wechsel nebeneinander", value=True, key="e1_alle")
    lupe_an = e1c.checkbox("Lupe zeigen", value=True, key="e1_lupe")
    if hand or n > 50:
        st.info("Die Live-Rechnung dieses Experiments gilt für n = 10, 20 und 50. Die eingefrorene Tabelle unten enthält die Wechselzellen bei n = 20 und 50.")
    else:
        m_eff = KO.cap_live(n, (zeiger, merkmal), seeds_zahl, m_wunsch)
        if m_eff < m_wunsch:
            st.caption(f"Rechengrenze: höchstens {V.zahl(C.LIVE_O_MAX)} grobe Operationen je Klick, darum {m_eff} statt {m_wunsch} Instanzen.")
        w_liste = ["basis", "klein", "gross", "versch"] if alle_w else ["basis"]
        if not wechsel_ist_basis and not any(
            (w == "klein" and (faktor, vx, vy) == (C.FAKTOR_KLEIN, 0, 0)) or (w == "gross" and (faktor, vx, vy) == (C.FAKTOR_GROSS, 0, 0)) or (w == "versch" and (faktor, vx, vy) == (1.0, 3, 2)) for w in w_liste):
            w_liste.append((faktor, vx, vy))
        with st.spinner(f"Rechne {m_eff} Instanzen × 2 Netze × {seeds_zahl} Zufalls-Seeds × {len(w_liste)} Zustände ..."):
            zellen = _e1_live(familie, n, m_eff, seeds_zahl, zeiger, merkmal, tuple(tuple(w) if not isinstance(w, str) else w for w in w_liste))
        if any(v is None for v in zellen.values()):
            st.info("Für diese Instanzen liegt keine Referenz vor.")
        else:
            def beschr(w):
                return C.WECHSEL_ANZEIGE[w] if isinstance(w, str) else f"Seitenleiste: Faktor {V.de(w[0], 2)}, ({w[1]}; {w[2]})"

            zeilen = []
            for w in w_liste:
                for netz in (zeiger, merkmal):
                    z = zellen[(w if isinstance(w, str) else tuple(w), netz)]
                    zeilen.append({"wechsel": beschr(w), "netz": C.NETZ_KURZ[netz], "v": z["v"], "lo": z["lo"], "hi": z["hi"], "farbe": C.NETZ_FARBE[netz]})
            st.plotly_chart(VIS.fig_wechsel_balken(zeilen), key="e1_balken", width="stretch")
            if wechsel_ist_basis:
                w_urteil = "basis"
            else:
                w_urteil = next((w for w in w_liste if (not isinstance(w, str))), None)
                if w_urteil is None:
                    w_urteil = "klein" if (faktor, vx, vy) == (C.FAKTOR_KLEIN, 0, 0) else "gross" if (faktor, vx, vy) == (C.FAKTOR_GROSS, 0, 0) else "versch"
            erg = [{"name": C.NETZ_KURZ[netz], "z": zellen[(w_urteil if isinstance(w_urteil, str) else tuple(w_urteil), netz)], "invariant": netz in C.INVARIANTE_NETZE and not wechsel_ist_basis} for netz in (zeiger, merkmal)]
            zeige(V.verdict_skala(faktor, vx, vy, erg, m_eff, V.kosten_zeile(n, kosten_n["K"], kosten_n["f_zeiger"], kosten_n["f_merkmal"])))

    if fr is not None:
        st.markdown("**Eingefrorene Tabelle der 18 Wechselzellen** (Training bei n = 20 gleichverteilt, je Zelle Lücke zur Referenz / Unterschied zum Nächsten Nachbarn in Prozentpunkten, **fett** = bricht):")
        netze_tab = [x for x in C.NETZE_BASIS if fr.hat_netz(x)]
        kopf = "| Zelle | " + " | ".join(C.NETZ_KURZ[x] for x in netze_tab) + " |"
        zl = [kopf, "|---|" + "---|" * len(netze_tab)]
        for nn_ in C.WECHSEL_GROESSEN:
            for fam in C.FAMILIEN:
                for art in ("klein", "gross", "versch"):
                    zs = []
                    for netz in netze_tab:
                        z = fr.zelle(fam, nn_, netz, art)
                        txt = f"{V.de(z['luecke'], 1)} / {V.de_vz(z['v'], 1)}"
                        zs.append(f"**{txt}**" if z["klasse"] == "bricht" else txt)
                    zl.append(f"| n = {nn_}, {C.FAMILIE_ANZEIGE[fam]}, {C.WECHSEL_ANZEIGE[art]} | " + " | ".join(zs) + " |")
        with st.expander("Tabelle der 18 Wechselzellen anzeigen"):
            st.markdown("\n".join(zl))
        zaehl = []
        for netz in netze_tab:
            k = V.zaehle_klassen(fr.wechsel_klassen(netz).values())
            zaehl.append(f"{C.NETZ_KURZ[netz]} {k['bricht']} von {k['gesamt']}")
        st.markdown("Zellen mit Bruch: " + ", ".join(zaehl) + ". Die Gleichheit mit der Basiszelle bei A+nt, An und B mit s ist Konstruktion (die Netze sehen dieselbe Eingabe), kein Befund über Lernen. "
                    f"Das Zeigernetz A bleibt in keiner der Wechselzellen auf der Lücke seiner Basiszelle (Anstieg unter {V.de(C.MARKE_HAELT_LUECKE, 0)} Prozentpunkten): in {fr.wechsel_luecke_gleich_basis('A')} von 18 Zellen steigt sie um mindestens {V.de(C.MARKE_HAELT_LUECKE, 0)} Prozentpunkte.")

    if lupe_an:
        st.markdown(TX.LUPE_TEXT)
        hand_x = HA.PUNKTE
        zl = ["| Zustand | Zeiger-Werte u der drei freien Orte P2, P3, P4 | Spannweite |", "|---|---|---|"]
        for name, xx in (("Basis", hand_x), ("Faktor 0,25", 0.25 * hand_x), ("Faktor 10", 10.0 * hand_x), ("Verschiebung (3; 2)", hand_x + np.array([3.0, 2.0]))):
            u = HA.zeiger_logits_spielzeug(xx, 1, [2, 3, 4])
            zl.append(f"| {name} | {V.de(u[0], 3)}; {V.de(u[1], 3)}; {V.de(u[2], 3)} | {V.wiss(HA.spannweite(u))} |")
        st.markdown("Spielzeug-Zeigerstufe an den fünf Orten des Handbeispiels (aktuelle Stadt P1; kein trainiertes Netz):\n\n" + "\n".join(zl))
        if not hand and MO.vorhanden("A") and n > 50:
            st.caption("Die Diagnose an der gewählten Instanz rechnet live bis n = 50; für größere n gilt die eingefrorene Tabelle (n = 20).")
        if not hand and MO.vorhanden("A") and n <= 50:
            seeds_inst = tuple(range(min(C.DIAGNOSE_INSTANZEN, 10)))
            sat, kap = _diagnose_live(familie, n, seeds_inst, faktor, vx, vy, seeds_zahl)
            dg = fr.d.get("diagnose") if fr is not None else None
            st.markdown(f"**Diagnose an den Instanzen 0 bis {len(seeds_inst) - 1} im gewählten Wechsel** (Zeigernetz A, Merkmalsnetz B ohne s, Mittel über {seeds_zahl} Zufalls-Seeds): "
                        f"Anteil der Zeiger-Werte im Sättigungsbereich (|u| > 9,64 von 10) unter den freien Orten **{pct(100 * sat)}**; Anteil gekappter Entfernungsmerkmale von B ohne s: "
                        f"x1 {pct(100 * kap['x1'])}, x3 {pct(100 * kap['x3'])}, x4 {pct(100 * kap['x4'])}, x5 gleich 0 bei {pct(100 * kap['x5_null'])}. **Das sind Diagnosen, keine Ursachenmessung.**")
            if dg:
                zl = ["| Zustand (n = 20 gleichverteilt, eingefroren) | Sättigungsanteil A | gekappt x1 (B ohne s) | gekappt x3 | gekappt x4 | x5 gleich 0 |", "|---|---|---|---|---|---|"]
                for art in C.WECHSEL:
                    zl.append(f"| {C.WECHSEL_ANZEIGE[art]} | {pct(100 * dg['saettigung'][art]['mittel'])} | {pct(100 * dg['kappung_bohne'][art]['x1'])} | {pct(100 * dg['kappung_bohne'][art]['x3'])} | "
                              f"{pct(100 * dg['kappung_bohne'][art]['x4'])} | {pct(100 * dg['kappung_bohne'][art]['x5_null'])} |")
                st.markdown("\n".join(zl))
            tour_basis = EV.touren_zeiger("A", MO.lade("A", netz_seed)[0], X[None])[0]
            t_schritt = st.slider("Schritt t im Zeigernetz A (erstes Modell)", 1, n - 1, 1, key=f"e1_lupe_t_{n}", help="Wahrscheinlichkeit, mit der das Netz im Schritt t auf jeden noch freien Ort zeigt, auf der Basisinstanz und im gewählten Wechsel (gleiche Tour).")
            P0 = MO.lade("A", netz_seed)[0]
            ps_b = N.probabilities_along(P0, X, tour_basis)
            ps_w = N.probabilities_along(P0, Xw, tour_basis)
            frei = [int(j) for j in tour_basis[t_schritt:]]
            st.plotly_chart(VIS.fig_wahrscheinlichkeit(ps_b[t_schritt - 1], None if wechsel_ist_basis else ps_w[t_schritt - 1], frei, f"Wahrscheinlichkeiten im Schritt {t_schritt} (Zeigernetz A, rohe Eingabe)",
                                                      f"Faktor {V.de(faktor, 2)}, ({vx}; {vy})"), key="e1_lupe_p", width="stretch")

# ---------------------------------------------------------------------------------------------------------------- E2 Größe
st.subheader(TX.E2_TITEL)
st.caption(TX.E2_NUTZT)
if fr is not None:
    e2a, e2b = st.columns(2)
    alle_netze = e2a.checkbox("alle Netzvarianten zeigen", value=False, key="e2_alle")
    zeige_2opt = e2b.checkbox("Nächster Nachbar mit 2-opt zeigen", value=True, key="e2_2opt")
    netze_e2 = [x for x in C.NETZE_BASIS if fr.hat_netz(x)] if alle_netze else [x for x in (zeiger, merkmal) if fr.hat_netz(x)]
    zl = []
    for netz in netze_e2:
        for g in C.GROESSEN:
            z = fr.zelle(familie, g, netz)
            zl.append({"netz": C.NETZ_KURZ[netz], "n": g, "v": z["v"], "lo": z["lo"], "hi": z["hi"], "farbe": C.NETZ_FARBE[netz], "m": z["m"]})
    st.plotly_chart(VIS.fig_groesse(zl, f"Unterschied zum Nächsten Nachbarn, Familie {C.FAMILIE_ANZEIGE[familie]}"), key="e2_kurve", width="stretch")
    kopf = "| Lücke zur Referenz in Prozent | " + " | ".join(f"n = {g}" for g in C.GROESSEN) + " |"
    tab = [kopf, "|---|" + "---|" * len(C.GROESSEN), "| Nächster Nachbar | " + " | ".join(V.de(fr.luecke_nn(familie, g), 1) for g in C.GROESSEN) + " |"]
    if zeige_2opt:
        tab.append("| Nächster Nachbar + 2-opt | " + " | ".join((V.de(fr.luecke_nn2opt(familie, g), 1) if fr.luecke_nn2opt(familie, g) is not None else "–") for g in C.GROESSEN) + " |")
    for netz in netze_e2:
        tab.append(f"| {C.NETZ_KURZ[netz]} | " + " | ".join(V.de(fr.luecke_netz(familie, g, netz), 1) for g in C.GROESSEN) + " |")
    tab.append("| Instanzen je Zelle | " + " | ".join(str(fr.m(g)) for g in C.GROESSEN) + " |")
    tab.append("| K je Konstruktion | " + " | ".join(V.zahl(KO.k_konstruktion(g)) for g in C.GROESSEN) + " |")
    tab.append("| O Nächster Nachbar / Zeigernetz / Merkmalsnetz (Millionen) | " + " | ".join(
        f"{V.de(KO.ops_nn(g) / 1e6, 3)} / {V.de(KO.ops_zeiger(g) / 1e6, 2)} / {V.de(KO.ops_merkmal(g) / 1e6, 2)}" for g in C.GROESSEN) + " |")
    tab.append("| s der Instanz geteilt durch s0 | " + " | ".join(V.de(fr.d["diagnose"]["s_verhaeltnis"][f"{familie}_{g}"], 2) for g in C.GROESSEN) + " |")
    st.markdown("\n".join(tab))
    gz = {g: fr.zelle(familie, g, zeiger)["klasse"] for g in C.GROESSEN}
    gm = {g: fr.zelle(familie, g, merkmal)["klasse"] for g in C.GROESSEN}
    ng = n if n in C.GROESSEN else 50
    g2 = fr.luecke_nn2opt(familie, ng)
    zeige(V.verdict_groesse(C.FAMILIE_ANZEIGE[familie], f"Zeigernetz ({C.NETZ_KURZ[zeiger]})", f"Merkmalsnetz ({C.NETZ_KURZ[merkmal]})", gz, gm, {g: fr.m(g) for g in C.GROESSEN}, ng, g2,
                            fr.luecke_netz(familie, ng, merkmal), V.kosten_zeile(ng, KO.k_konstruktion(ng), KO.kosten_je_konstruktion(ng)["f_zeiger"], KO.kosten_je_konstruktion(ng)["f_merkmal"])))
    st.caption("Ausblick, nicht in der Tabelle: ein bei n = 50 direkt trainiertes Zeigernetz erreichte in der früheren Messreihe (200 Instanzen, anderes Training, andere Instanzen) 19,72 % Lücke gegen 20,82 % für das bei n = 20 trainierte, ohne nachweisbaren Unterschied; Quelle: README.")
    if not hand:
        m_live = st.slider("Instanzen für den Einzelknopf", 1, C.E2_KNOPF_M_MAX, 2, key="e2_m", help="Die Zelle der Seitenleiste (Familie, n) mit diesen Instanzen und den gewählten Netzen; nur Anzeige, kein Urteil.")
        m_eff2 = KO.cap_live(n, (zeiger, merkmal), seeds_zahl, m_live, m_max=C.E2_KNOPF_M_MAX)
        if m_eff2 < m_live:
            st.caption(f"Rechengrenze: höchstens {V.zahl(C.LIVE_O_MAX)} grobe Operationen je Klick, darum {m_eff2} statt {m_live} Instanzen.")
        if st.button("Diese Größe jetzt live rechnen", key="e2_live"):
            with st.spinner(f"Rechne {m_eff2} Instanzen × 2 Netze × {seeds_zahl} Zufalls-Seeds (n = {n}) ..."):
                lz = _e2_live(familie, n, m_eff2, seeds_zahl, zeiger, merkmal)
            for netz in (zeiger, merkmal):
                z = lz[netz]
                st.markdown(f"{C.NETZ_KURZ[netz]}: mittlere Lücke {pct(z['luecke'])}, Unterschied zum Nächsten Nachbarn {V.de_vz(z['v'], 2)} Prozentpunkte ({m_eff2} Instanzen, nur Anzeige, kein Urteil)." if z else f"{C.NETZ_KURZ[netz]}: keine Referenz.")
    else:
        st.caption("Beim Handbeispiel (n = 5) gibt es keine Netzläufe.")

with st.expander("Kosten je Konstruktion (K und O, Training als dritter Posten)", expanded=False):
    kz = ["| n | K (alle drei gleich) | O Nächster Nachbar | O Zeigernetz | Faktor gegen Nächsten Nachbarn | O Merkmalsnetz | Faktor gegen Nächsten Nachbarn |", "|---|---|---|---|---|---|---|"]
    for g in C.GROESSEN:
        k = KO.kosten_je_konstruktion(g)
        kz.append(f"| {g} | {V.zahl(k['K'])} | {V.zahl(k['O_nn'])} | {V.zahl(k['O_zeiger'])} | {V.faktor(k['f_zeiger'])} | {V.zahl(k['O_merkmal'])} | {V.faktor(k['f_merkmal'])} |")
    st.markdown("\n".join(kz))
    st.markdown(f"Training als dritter Posten: Zeigernetz {V.zahl(C.TOUREN_ZEIGER)} gesampelte Touren, Merkmalsnetz {V.zahl(C.TOUREN_MERKMAL)} Touren und {V.zahl(C.ZEILEN_MERKMAL)} Merkmalszeilen. "
                "K begünstigt die Netze (ein Zeigerschritt oder eine Merkmalszeile kostet mehr als ein Vergleich), O benachteiligt sie (Annahme 8 Operationen je Vergleich). Ab n = 100 kostet das Merkmalsnetz in O mehr als das Zeigernetz.")

# ---------------------------------------------------------------------------------------------------------------- E3 Verteilung
st.subheader(TX.E3_TITEL)
st.caption(TX.E3_NUTZT)
if fr is not None:
    n3 = n if n in C.GROESSEN else 20
    if hand:
        st.caption("Beim Handbeispiel (n = 5) zeigt dieses Experiment n = 20.")
    gegenueber = st.checkbox("Normierung gegenüberstellen (A, A+nt, An nebeneinander)", value=True, key="e3_norm")
    netze3 = [x for x in (("A", "Ant", "An") if gegenueber else (zeiger,)) if fr.hat_netz(x)] + [x for x in ((merkmal,) if not gegenueber else ("Bmit", "Bohne")) if fr.hat_netz(x)]
    zl = []
    for netz in netze3:
        for fam in C.FAMILIEN:
            z = fr.zelle(fam, n3, netz)
            zl.append({"familie": C.FAMILIE_ANZEIGE[fam], "netz": C.NETZ_KURZ[netz], "v": z["v"], "lo": z["lo"], "hi": z["hi"], "farbe": C.NETZ_FARBE[netz], "seeds_pos": z["seeds_pos"], "seeds": z["seeds"]})
    st.plotly_chart(VIS.fig_familien(zl, f"Unterschied zum Nächsten Nachbarn bei n = {n3} je Familie"), key="e3_balken", width="stretch")
    cs = st.columns(3)
    for col, fam in zip(cs, C.FAMILIEN):
        Xf = TR.zufalls_instanz(fam, n3, 0)
        Df = TR.abstaende(Xf[None])
        col.plotly_chart(VIS.fig_karte(Xf, TR.nn_touren(Df)[0], f"{C.FAMILIE_ANZEIGE[fam]} (Zufalls-Seed 0, Nächster Nachbar)", C.COLOR_NN, hoehe=260), key=f"e3_karte_{fam}", width="stretch")
    ak = {f: fr.zelle(f, n3, "A")["klasse"] for f in C.FAMILIEN}
    bk = {f: fr.zelle(f, n3, merkmal)["klasse"] for f in C.FAMILIEN}
    a_roh = {f: fr.zelle(f, n3, "A")["v"] for f in C.FAMILIEN}
    a_norm = {f: fr.zelle(f, n3, "Ant")["v"] for f in C.FAMILIEN}
    kk3 = KO.kosten_je_konstruktion(n3)
    ank = {f: fr.zelle(f, n3, "Ant")["klasse"] for f in C.FAMILIEN}
    zeige(V.verdict_verteilung(n3, "Zeigernetz (A)", f"Merkmalsnetz ({C.NETZ_KURZ[merkmal]})", {C.FAMILIE_ANZEIGE[f]: ak[f] for f in ak}, {C.FAMILIE_ANZEIGE[f]: bk[f] for f in bk},
                               {C.FAMILIE_ANZEIGE[f]: a_roh[f] for f in a_roh}, {C.FAMILIE_ANZEIGE[f]: a_norm[f] for f in a_norm}, V.kosten_zeile(n3, kk3["K"], kk3["f_zeiger"], kk3["f_merkmal"]),
                               {C.FAMILIE_ANZEIGE[f]: ank[f] for f in ank}))

# ---------------------------------------------------------------------------------------------------------------- E4 Stichprobe
st.subheader(TX.E4_TITEL)
st.caption(TX.E4_NUTZT)
if fr is not None:
    f1, f2, f3, f4, f5 = st.columns(5)
    n4 = f1.selectbox("Größe n der Zelle", options=C.GROESSEN, index=C.GROESSEN.index(200), key="e4_n")
    fam4 = f2.selectbox("Familie der Zelle", options=C.FAMILIEN, format_func=C.FAMILIE_ANZEIGE.get, key="e4_fam")
    netz_art = f3.radio("Modell", options=("zeiger", "merkmal"), format_func=lambda x: "Zeigernetz" if x == "zeiger" else "Merkmalsnetz", key="e4_netz")
    M4 = fr.m(n4)
    m4 = f4.slider("Teilstichprobe m", 3, max(4, M4), min(C.DEFAULT_E4_M, M4), key=f"e4_m_{n4}")
    R4 = f5.slider("Wiederholungen R", C.E4_R_MIN, C.E4_R_MAX, C.E4_R_STANDARD, key="e4_r")
    netz4 = zeiger if netz_art == "zeiger" else merkmal
    zc = fr.zelle(fam4, n4, netz4)
    delta = fr.delta_je_seed(fam4, n4, netz4)
    st.markdown(f"Zelle: {C.NETZ_ANZEIGE[netz4]}, n = {n4}, {C.FAMILIE_ANZEIGE[fam4]}; Vollstichprobe {M4} Instanzen: Urteil **{V.KLASSE_KURZ[zc['klasse']]}**, Unterschied {V.zellen_text(zc)}.")
    if m4 >= M4:
        zeige(V.verdict_stichprobe(m4, 0, 0, M4, V.KLASSE_KURZ[zc["klasse"]], 0.0, n4))
    else:
        R_eff = EX.e4_cap(m4, R4)
        if R_eff < R4:
            st.caption(f"Rechengrenze: R · {C.E4_BOOT} · m höchstens {V.zahl(C.E4_ZIEHUNGEN_MAX)} Ziehungen, darum R = {R_eff} statt {R4}.")
        res = EX.teilstichproben(delta, m4, R_eff, schluessel=n4 * 10 + C.FAMILIEN.index(fam4))
        gleich = sum(1 for k in res["klassen"] if k == zc["klasse"])
        zeige(V.verdict_stichprobe(m4, gleich, res["R"], M4, V.KLASSE_KURZ[zc["klasse"]], res["breite"], n4))
        st.plotly_chart(VIS.fig_streifen(res["v"], zc["v"], zc["lo"], zc["hi"], f"Mittel von {res['R']} Teilstichproben zu je {m4} Instanzen"), key="e4_streifen", width="stretch")
        grid = [g for g in (3, 5, 10, 20, 30, 50, 100) if g < M4]
        breiten = [EX.teilstichproben(delta, g, EX.e4_cap(g, 60), schluessel=7)["breite"] for g in grid]
        st.plotly_chart(VIS.fig_breite(grid, breiten), key="e4_breite", width="stretch")
        st.caption(f"Die Teilstichproben aus {M4} Instanzen sagen etwas über {m4} aus {M4} aus, nicht über {m4} aus der Grundgesamtheit der erzeugten Instanzen.")

# ---------------------------------------------------------------------------------------------------------------- E5 Mischtraining
st.subheader(TX.E5_TITEL)
st.caption(TX.E5_NUTZT)
if fr is not None and fr.hat_netz("Aaug") and fr.hat_netz("Amix"):
    k_basis_ant = V.zaehle_klassen(fr.basis_klassen("Ant").values())
    k_basis_mix = V.zaehle_klassen(fr.basis_klassen("Amix").values())
    k_basis_aug = V.zaehle_klassen(fr.basis_klassen("Aaug").values())
    k_basis_b = V.zaehle_klassen(fr.basis_klassen("Bmit").values())
    k_gross_mix = V.zaehle_klassen(fr.basis_klassen("Amix", 100).values())
    k_wech_aug = V.zaehle_klassen(fr.wechsel_klassen("Aaug").values())
    k_wech_a = V.zaehle_klassen(fr.wechsel_klassen("A").values())
    k200_mix = V.zaehle_klassen(v for (f_, g_), v in fr.basis_klassen("Amix").items() if g_ == 200)
    zl = ["| Variante | Zufalls-Seeds unter dem Nächsten Nachbarn bei n = 20 gleichverteilt (Modellgüte) | Basiszellen mit Bruch (15) | davon ab n = 100 (6) | Wechselzellen mit Bruch (18) |", "|---|---|---|---|---|"]
    for name, netz in (("A (rohe Eingabe)", "A"), ("A+nt (Normierung beim Einsatz)", "Ant"), ("Aaug (augmentiert trainiert)", "Aaug"), ("Amix (normiert, n = 10, 20, 50 gemischt)", "Amix"), ("B mit s", "Bmit")):
        kb = V.zaehle_klassen(fr.basis_klassen(netz).values())
        kg = V.zaehle_klassen(fr.basis_klassen(netz, 100).values())
        kw = V.zaehle_klassen(fr.wechsel_klassen(netz).values())
        ku = fr.seeds_unter_nn(netz)
        zl.append(f"| {name} | {ku} von {C.NETZ_SEEDS}" + (" (schwach)" if ku < C.SEEDS_MIN_BRUCH else "") + f" | {kb['bricht']} | {kg['bricht']} | {kw['bricht']} |")
    st.markdown("\n".join(zl))
    aaug_seeds = fr.seeds_unter_nn("Aaug")
    zeige(V.verdict_mischtraining(k_basis_mix["bricht"], k_basis_ant["bricht"], k_gross_mix["bricht"], k_gross_mix["gesamt"], k_wech_aug["bricht"], k_wech_aug["gesamt"], k200_mix["bricht"], k200_mix["gesamt"],
                                  V.ursache_satz(k_gross_mix["bricht"], k_wech_aug["bricht"], aaug_seeds < C.SEEDS_MIN_BRUCH), aaug_seeds))
    st.caption("Schwellen für den Satz zur Ursache (vor der Messung gesetzt, nicht abgeleitet): „am Training“ nur, wenn Amix ab n = 100 in höchstens 1 von 6 Zellen bricht und Aaug in höchstens 3 von 18 Wechselzellen; "
               "„an der Eingabedarstellung“ nur, wenn Amix ab n = 100 in mindestens 4 von 6 Zellen bricht; dazwischen „nicht entschieden“. Training in der App entfällt.")
else:
    st.info("Mischtraining: nicht gemessen.")

# ---------------------------------------------------------------------------------------------------------------- Grenzen, Annahmen
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(TX.ASSUMPTIONS_MARKDOWN)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(TX.FORMEL_LESEHILFE)
    for text, formel in TX.FORMELN:
        if text:
            st.markdown(text)
        st.latex(formel)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Neuronale kombinatorische Optimierung](https://sebastianhanisch.net/konzepte-neuronale-kombinatorische-optimierung.html)."
)

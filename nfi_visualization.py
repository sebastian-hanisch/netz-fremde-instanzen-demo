"""Plotly-Figuren (alle Achsen fest, `fixedrange`; Legenden unterhalb mit Rand; keine Beschriftung „outside“ an Balken in Containern ohne Größe)."""

import numpy as np
import plotly.graph_objects as go

import nfi_constants as C

HOEHE = 340


def _fest(fig: go.Figure, hoehe: int = HOEHE, legende: bool = True, unten: int = 70) -> go.Figure:
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(
        height=hoehe, margin=dict(l=50, r=15, t=40, b=unten if legende else 40), showlegend=legende, template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0.0), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def fig_karte(X: np.ndarray, tour, titel: str, farbe: str, schritt: int | None = None, hoehe: int = 330) -> go.Figure:
    """Karte einer Tour im Einheitsquadrat (gleicher Maßstab über unsichtbare Eckpunkte), Pfeile für die Kanten bis zum Schritt `schritt` (Standard: ganze Tour), Start Stadt 0 hervorgehoben."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    t = list(tour)
    bis = n if schritt is None else max(0, min(int(schritt), n))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[-0.05, 1.05], y=[-0.05, 1.05], mode="markers", marker=dict(opacity=0), hoverinfo="skip", showlegend=False))
    if bis >= 1:
        kanten = t[:bis + 1] if bis < n else t + [t[0]]
        fig.add_trace(go.Scatter(x=X[kanten, 0], y=X[kanten, 1], mode="lines", line=dict(color=farbe, width=2), hoverinfo="skip", showlegend=False))
        if n <= 60:                     # Pfeile nur bei überschaubarer Zahl (je Annotation kostet Plotly Zeit; bei n = 200 wären es 200)
            fig.update_layout(annotations=[dict(x=X[b, 0], y=X[b, 1], ax=X[a, 0], ay=X[a, 1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.0, arrowwidth=1.5,
                                                arrowcolor=farbe, standoff=3, text="") for a, b in zip(kanten[:-1], kanten[1:])])
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", marker=dict(size=7, color="#444"), hovertemplate="Ort %{pointindex}<extra></extra>", showlegend=False))
    fig.add_trace(go.Scatter(x=[X[t[0], 0]], y=[X[t[0], 1]], mode="markers", marker=dict(size=13, color=farbe, symbol="star", line=dict(color="#222", width=1)), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(title=dict(text=titel, x=0.02, font=dict(size=14)))
    return _fest(fig, hoehe, legende=False)


def fig_wechsel_balken(zeilen: list, titel: str = "Unterschied zum Nächsten Nachbarn je Wechsel (Prozentpunkte der Referenz)") -> go.Figure:
    """Gruppierte Balken. `zeilen`: Liste von {"wechsel", "netz", "v", "lo", "hi", "farbe"}; negative Werte: Netz kürzer als der Nächste Nachbar; Intervall 95 % Bootstrap über Instanzen."""
    fig = go.Figure()
    netze = []
    for z in zeilen:
        if z["netz"] not in netze:
            netze.append(z["netz"])
    wechsel = []
    for z in zeilen:
        if z["wechsel"] not in wechsel:
            wechsel.append(z["wechsel"])
    for netz in netze:
        zz = {z["wechsel"]: z for z in zeilen if z["netz"] == netz}
        fig.add_trace(go.Bar(
            name=netz, x=wechsel, y=[zz[w]["v"] if w in zz else None for w in wechsel], marker_color=next(z["farbe"] for z in zeilen if z["netz"] == netz),
            error_y=dict(type="data", symmetric=False, array=[(zz[w]["hi"] - zz[w]["v"]) if (w in zz and zz[w]["hi"] is not None) else 0 for w in wechsel],
                         arrayminus=[(zz[w]["v"] - zz[w]["lo"]) if (w in zz and zz[w]["lo"] is not None) else 0 for w in wechsel]),
            hovertemplate="%{x}: %{y:+.2f} Prozentpunkte<extra>" + netz + "</extra>"))
    fig.add_hline(y=0, line_width=3, line_color="#222")
    fig.update_layout(title=dict(text=titel, x=0.02, font=dict(size=14)), barmode="group")
    fig.update_yaxes(title="Prozentpunkte (negativ: Netz besser)", zeroline=False)
    return _fest(fig, 360)


def fig_groesse(zeilen: list, titel: str = "Unterschied zum Nächsten Nachbarn gegen die Zahl der Orte n") -> go.Figure:
    """Linien mit Intervallbalken. `zeilen`: Liste von {"netz", "n", "v", "lo", "hi", "farbe", "m"} (m: Instanzen); n logarithmisch."""
    fig = go.Figure()
    netze = []
    for z in zeilen:
        if z["netz"] not in netze:
            netze.append(z["netz"])
    for netz in netze:
        zz = sorted((z for z in zeilen if z["netz"] == netz), key=lambda z: z["n"])
        fig.add_trace(go.Scatter(
            name=netz, x=[z["n"] for z in zz], y=[z["v"] for z in zz], mode="lines+markers", line=dict(color=zz[0]["farbe"], width=2), marker=dict(size=8),
            error_y=dict(type="data", symmetric=False, array=[(z["hi"] - z["v"]) if z["hi"] is not None else 0 for z in zz], arrayminus=[(z["v"] - z["lo"]) if z["lo"] is not None else 0 for z in zz]),
            customdata=[z["m"] for z in zz], hovertemplate="n = %{x}, %{y:+.2f} Prozentpunkte, %{customdata} Instanzen<extra>" + netz + "</extra>"))
    fig.add_hline(y=0, line_width=3, line_color="#222")
    fig.update_xaxes(type="log", tickvals=list(C.GROESSEN), ticktext=[str(n) for n in C.GROESSEN], title="Orte n (Training bei n = 20)")
    fig.update_yaxes(title="Prozentpunkte (negativ: Netz besser)")
    fig.update_layout(title=dict(text=titel, x=0.02, font=dict(size=14)))
    return _fest(fig, 380)


def fig_familien(zeilen: list, titel: str = "Unterschied zum Nächsten Nachbarn je Familie") -> go.Figure:
    """Gruppierte Balken je Familie. `zeilen`: Liste von {"familie", "netz", "v", "lo", "hi", "farbe", "seeds_pos", "seeds"}."""
    fig = go.Figure()
    netze, fams = [], []
    for z in zeilen:
        if z["netz"] not in netze:
            netze.append(z["netz"])
        if z["familie"] not in fams:
            fams.append(z["familie"])
    for netz in netze:
        zz = {z["familie"]: z for z in zeilen if z["netz"] == netz}
        fig.add_trace(go.Bar(
            name=netz, x=fams, y=[zz[f]["v"] if f in zz else None for f in fams], marker_color=next(z["farbe"] for z in zeilen if z["netz"] == netz),
            error_y=dict(type="data", symmetric=False, array=[(zz[f]["hi"] - zz[f]["v"]) if f in zz else 0 for f in fams], arrayminus=[(zz[f]["v"] - zz[f]["lo"]) if f in zz else 0 for f in fams]),
            customdata=[f"{zz[f]['seeds_pos']} von {zz[f]['seeds']} Zufalls-Seeds über null" if f in zz else "" for f in fams],
            hovertemplate="%{x}: %{y:+.2f} Prozentpunkte, %{customdata}<extra>" + netz + "</extra>"))
    fig.add_hline(y=0, line_width=3, line_color="#222")
    fig.update_layout(title=dict(text=titel, x=0.02, font=dict(size=14)), barmode="group")
    fig.update_yaxes(title="Prozentpunkte (negativ: Netz besser)")
    return _fest(fig, 360)


def fig_streifen(v: np.ndarray, v_voll: float, lo_voll: float | None, hi_voll: float | None, titel: str) -> go.Figure:
    """Streifendiagramm der Mittel der Teilstichproben (E4); Linie: Mittel der Vollstichprobe, Band: deren Intervall; Null fett."""
    rng = np.arange(len(v))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.sort(v), y=rng, mode="markers", marker=dict(size=5, color=C.COLOR_B, opacity=0.6), name="Mittel einer Teilstichprobe", hovertemplate="%{x:+.2f} Prozentpunkte<extra></extra>"))
    if lo_voll is not None and hi_voll is not None:
        fig.add_vrect(x0=lo_voll, x1=hi_voll, fillcolor=C.COLOR_BAND, line_width=0, annotation_text="Intervall der Vollstichprobe", annotation_position="top left")
    fig.add_vline(x=v_voll, line_color=C.COLOR_A, line_width=2)
    fig.add_vline(x=0, line_color="#222", line_width=3)
    fig.update_xaxes(title="Mittlerer Unterschied einer Teilstichprobe (Prozentpunkte)")
    fig.update_yaxes(title="Teilstichproben (sortiert)", showticklabels=False)
    fig.update_layout(title=dict(text=titel, x=0.02, font=dict(size=14)))
    return _fest(fig, 360)


def fig_breite(ms: list, breiten: list, titel: str = "Breite des Intervalls gegen die Zahl der Instanzen") -> go.Figure:
    """Mittlere halbe Intervallbreite gegen die Teilstichprobe m."""
    fig = go.Figure(go.Scatter(x=ms, y=breiten, mode="lines+markers", line=dict(color=C.COLOR_B), name="halbe Breite"))
    fig.update_xaxes(title="Instanzen m", type="log")
    fig.update_yaxes(title="halbe Breite (Prozentpunkte)", rangemode="tozero")
    fig.update_layout(title=dict(text=titel, x=0.02, font=dict(size=14)))
    return _fest(fig, 320, legende=False)


def fig_wahrscheinlichkeit(p_basis: np.ndarray, p_wechsel: np.ndarray | None, frei: list, titel: str, name_wechsel: str = "gewählter Wechsel") -> go.Figure:
    """Wahrscheinlichkeiten p_t(j) der freien Orte in einem Zeigerschritt, auf der Basisinstanz und unter dem Wechsel."""
    x = [f"Ort {j}" for j in frei]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Basisinstanz", x=x, y=[float(p_basis[j]) for j in frei], marker_color=C.COLOR_A))
    if p_wechsel is not None:
        fig.add_trace(go.Bar(name=name_wechsel, x=x, y=[float(p_wechsel[j]) for j in frei], marker_color=C.COLOR_NN))
    fig.update_layout(title=dict(text=titel, x=0.02, font=dict(size=14)), barmode="group")
    fig.update_yaxes(title="Wahrscheinlichkeit", range=[0, 1.0])
    return _fest(fig, 320)

import matplotlib
matplotlib.use("Agg")

import re
import os

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from collections import defaultdict


EVENT_RE = re.compile(
    r"""
    \(\s*
        (?P<head>e\d+|\d+)\s*,\s*
        (?P<rel>\d+)\s*,\s*
        (?P<tail>e\d+|\d+)\s*,\s*
        (?P<tvar>t\d+)
        (?:\s*=\s*
            (?P<tref>t\d+)
            \s*\+\s*
            \(\s*(?P<lag_min>-?\d+)\s*,\s*(?P<lag_max>-?\d+)\s*\)
        )?
    \s*\)
    """,
    flags=re.VERBOSE
)

@dataclass
class Event:
    role: str
    idx: int
    head: str
    rel: int
    tail: str
    tvar: str
    tref: Optional[str]
    lag: Optional[Tuple[int,int]]


def _parse_event(s: str, role: str, idx: int) -> Event:
    m = EVENT_RE.search(s)
    if not m:
        raise ValueError(f"Could not parse event: {s!r}")
    head = m.group("head")
    rel = int(m.group("rel"))
    tail = m.group("tail")
    tvar = m.group("tvar")
    tref = m.group("tref")
    lag = None
    if m.group("lag_min") is not None:
        lag = (int(m.group("lag_min")), int(m.group("lag_max")))
    return Event(role=role, idx=idx, head=head, rel=rel, tail=tail, tvar=tvar, tref=tref, lag=lag)

def parse_pattern_line(line: str) -> Dict:
    """Accepts the FIRST column (pattern text) if you pass a full TSV line."""
    pattern_text = line.split("\t")[0].strip()
    if "->" not in pattern_text:
        raise ValueError("Pattern line must contain '->'")
    left, right = pattern_text.split("->", 1)
    ants = [s.strip() for s in left.split("&") if s.strip()]
    antecedents = [_parse_event(s, role="a", idx=i) for i, s in enumerate(ants)]
    consequence = _parse_event(right.strip(), role="c", idx=0)

    def t_key(tv): return int(tv[1:]) if tv.startswith("t") and tv[1:].isdigit() else 10**9
    time_vars = sorted({ev.tvar for ev in antecedents + [consequence]}, key=t_key)

    lags = []
    for ev in antecedents + [consequence]:
        if ev.tref and ev.lag:
            lags.append((ev.tvar, ev.tref, ev.lag))

    entities = sorted(
        {ev.head for ev in antecedents + [consequence]} |
        {ev.tail for ev in antecedents + [consequence]},
        key=lambda x: (0, int(x[1:])) if x.startswith("e") else (1, int(x))
    )
    return {
        "text": pattern_text,
        "antecedents": antecedents,
        "consequence": consequence,
        "time_vars": time_vars,
        "lags": lags,
        "entities": entities,
    }

# Drawing helpers
def _color_for_tvar(tvar: str, palette=None):
    # Your preferred palette
    palette = palette or ['#67a9cf','#3690c0','#016c59','#014636']
    idx = int(tvar[1:]) if tvar.startswith("t") and tvar[1:].isdigit() else 0
    return palette[(idx - 1) % len(palette)]

def _draw_edge_labels_multigraph(G, pos, edge_labels, ax=None, offset_base=0.10):
    """Replacement that supports Multi(Di)Graph edge labels."""
    if ax is None:
        ax = plt.gca()

    group = defaultdict(list)   # (u,v) -> [k1,k2,...]
    for u, v, k in G.edges(keys=True):
        group[(u, v)].append(k)
    order = {(u, v, k): i for (u, v), keys in group.items() for i, k in enumerate(keys)}
    counts = {(u, v): len(keys) for (u, v), keys in group.items()}

    for (u, v, k), label in edge_labels.items():
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        if u == v:
            i = order[(u, v, k)]
            n = counts[(u, v)]
            dy = 0.20 + (i - (n - 1) / 2) * 0.06
            ax.text(x1, y1 + dy, label, fontsize=9,
                    ha="center", va="center",
                    bbox=dict(alpha=0.3, color="white", pad=1))
            continue

        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dx, dy = (x2 - x1), (y2 - y1)
        length = (dx**2 + dy**2) ** 0.5 or 1.0
        nxp, nyp = -dy / length, dx / length  # unit normal

        i = order[(u, v, k)]
        n = counts[(u, v)]
        offset = (i - (n - 1) / 2) * offset_base
        lx, ly = mx + nxp * offset, my + nyp * offset

        ax.text(lx, ly, label, fontsize=9,
                ha="center", va="center",
                bbox=dict(alpha=0.3, color="white", pad=1))

def _circular_positions(nodes: List[str], radius: float = 1.8, start_angle: float = np.pi/2):
    """Deterministic circular layout in the given node order."""
    N = max(1, len(nodes))
    pos = {}
    for i, n in enumerate(nodes):
        theta = start_angle - 2*np.pi * i / N  # clockwise from top
        pos[n] = (radius * np.cos(theta), radius * np.sin(theta))
    return pos


def plot_pattern(pattern_line: str, figsize=(7.8, 6.0), layout: str = "circular",
                 seed: int = 7, savepath: Optional[str] = None, title: Optional[str] = None,
                 palette: Optional[List[str]] = None):
    """Render ONE pattern (pattern column, or full TSV line) and optionally save to PNG."""
    data = parse_pattern_line(pattern_line)
    ants = data["antecedents"]
    cons = data["consequence"]
    tvars = data["time_vars"]
    entities = data["entities"]

    G = nx.MultiDiGraph()
    for e in entities:
        G.add_node(e)

    for ev in ants:
        G.add_edge(ev.head, ev.tail, key=f"a{ev.idx}", role="a", tvar=ev.tvar, rel=ev.rel, label=f"{ev.rel}@{ev.tvar}")
    G.add_edge(cons.head, cons.tail, key="c", role="c", tvar=cons.tvar, rel=cons.rel, label=f"{cons.rel}@{cons.tvar}")

    # --- positions ---
    if layout == "circular":
        entities_sorted = sorted(entities, key=lambda x: (0, int(x[1:])) if x.startswith("e") else (1, int(x)))
        pos = _circular_positions(entities_sorted, radius=1.8, start_angle=np.pi/2)
    else:
        pos = nx.spring_layout(G, seed=seed, k=1.1/(1+len(G.nodes())**0.5))

    fig = plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color="#f6f6f6", edgecolors="#444", linewidths=1.2)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold")

    # Edges grouped by time variable for consistent color
    for tv in tvars:
        elist = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d.get("tvar") == tv]
        for (u, v, k) in elist:
            role = G.edges[u, v, k]["role"]
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)],
                connectionstyle="arc3,rad=0.22",
                arrows=True, arrowstyle="-|>", arrowsize=19,
                width=3.0 if role == "c" else 2.0,
                style="solid" if role == "c" else "dashed",
                edge_color=_color_for_tvar(tv, palette),
            )

    # Edge labels (MultiDiGraph-safe)
    edge_labels = {(u, v, k): d.get("label", "") for u, v, k, d in G.edges(keys=True, data=True)}
    _draw_edge_labels_multigraph(G, pos, edge_labels)

    # Legend
    handles = [plt.Line2D([0],[0], color=_color_for_tvar(tv, palette), lw=3, label=tv) for tv in tvars]
    if handles:
        plt.legend(handles=handles, title="Time steps", loc="upper left", bbox_to_anchor=(1.02, 1.0))

    # Time constraints box
    if data["lags"]:
        lines = [f"{t} = {ref} + ({a},{b})" for (t, ref, (a,b)) in data["lags"]]
        text = "Time constraints:\n" + "\n".join(lines)
        plt.gca().text(1.02, 0.5, text, transform=plt.gca().transAxes, va="center", ha="left",
                       bbox=dict(facecolor="white", edgecolor="#ccc", boxstyle="round,pad=0.5"))

    if title is None:
        title = data["text"]
    plt.title(title, fontsize=10, pad=10)
    plt.axis("off")
    plt.tight_layout()

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=220, bbox_inches="tight")
        plt.close(fig)  # important in loops
    else:
        # In scripts we don't usually show; still close to avoid leaks.
        plt.close(fig)

def render_patterns_file(pattern2id_path: str, out_dir: str,
                         layout: str = "circular",
                         palette: Optional[List[str]] = None) -> int:
    """
    Read a tab-separated pattern2id.txt (columns: pattern_text, n_hops, id)
    and write one PNG per pattern to out_dir named <id>.png.
    Returns the number of images successfully written.
    """
    os.makedirs(out_dir, exist_ok=True)
    ok = 0
    with open(pattern2id_path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            # Defensive split; tolerate extra columns
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                # skip blank or malformed rows
                continue
            pattern_text, n_hops_str, pid_str = parts[0], parts[1], parts[2]
            # basic sanity
            try:
                pid = int(pid_str)
            except Exception:
                # header or malformed line
                continue

            savepath = os.path.join(out_dir, f"{pid}.png")
            title = f"Pattern {pid} (hops={n_hops_str})\n{pattern_text}"
            try:
                plot_pattern(pattern_text, savepath=savepath, layout=layout, title=title, palette=palette)
                ok += 1
            except Exception as e:
                # Don't fail the whole run if one pattern can't be parsed/drawn
                print(f"[pattern_plotter] Skipping line {line_no} (id={pid}): {e}")
                continue
    return ok

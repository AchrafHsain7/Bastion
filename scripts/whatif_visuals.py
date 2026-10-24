"""
What-If Comparison Script — Dual-Specification Safety Game Analysis
====================================================================
Loads all bastion_{IDX}_summary.json files from ../results/
and produces the 4 paper figures, JAIR-camera-ready.

Data contract (UNCHANGED from the original — every value is read from JSON,
nothing is computed from assumptions or filled in by hand):
    Each summary.json must contain:
    - "results"     -> MARL metrics (reward, dominance, per_run, CIs)
    - "safety_game" -> attractor/shield metrics from the analyzer
    Case descriptions live in CASE_DESCRIPTIONS below.

Run AFTER all 5 case studies have completed.

JAIR polish (vs. the draft figures):
    * EMBED_TITLES=False -> embedded titles stripped; the LaTeX \\caption{} carries them.
    * All secondary text darkened from wispy gray to charcoal (survives B/W print).
    * fig1/fig2: axis scaling is a toggle, ABSOLUTE_AXES (default True). All four
      metrics are bounded in [0,1] by construction, so absolute mode needs no
      rescaling and yields portable fingerprints; relative mode restores the
      cross-case min-max normalization for maximum within-set separation.
    * fig2: |W| subtext enlarged/bolded/darkened + thousands separators; bold axis key.
    * fig1/fig2: four axes (WIN, STP, DDR, SLT), each in [0,1] and read by its own
      definition (STP = shell concentration). Static layer {WIN, STP, SLT} + adaptive layer {DDR}. Earlier
      Violation Proximity removed as algebraically derived / restated; see the
      metric review. fig3 and fig4 are unchanged.
    * fig3: x-axis capped at 100% (no dead space); |W| totals moved into the y labels.
    * fig4: case captions upright+dark, reference label darkened, CI bar made deliberate;
            jitter seeded so the figure is reproducible across rebuilds.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════
RESULTS_DIR = "../results"
OUTPUT_DIR  = "../paper_results"
CASE_INDICES = [1, 2, 3, 4, 5]

# Reproducible jitter for the dominance swarm (fig4). Seed -> identical figure every build.
RNG_SEED = 11

# ── Case descriptions — displayed on figures and table ──
CASE_DESCRIPTIONS = {
    1: "Baseline topology",
    2: "Fully connected topology",
    3: "Relaxed φ_A: Unlimited destroys",
    4: "Relaxed φ_D: active >= 2",
    5: "VPN bypass removed",
}

# ── Academic palette: colorblind-safe, high contrast on white ──
COLORS = {
    1: "#2C5F8A",   # deep blue     — baseline (authoritative)
    2: "#D4652F",   # burnt orange  — full connectivity
    3: "#1B998B",   # teal          — relaxed phi_A
    4: "#A4243B",   # crimson       — relaxed phi_D
    5: "#5C4D7D",   # slate purple  — no VPN bypass
}
BG_COLOR     = "#FFFFFF"
GRID_COLOR   = "#E0E0E0"
TEXT_COLOR   = "#2D2D2D"
MUTED_COLOR  = "#33373B"   # dark charcoal for SECONDARY text — survives B/W print
ACCENT_COLOR = "#2C5F8A"
SPINE_COLOR  = "#BDBDBD"

# JAIR mode: figure captions carry the title, so strip embedded suptitles.
# Flip to True for slides / talks where a self-contained title helps.
EMBED_TITLES = False

# Fingerprint radius scaling.
#   True  — ABSOLUTE. All four axes are bounded in [0,1] by construction, so the
#           rim is 1 on every axis and no rescaling is applied. A fingerprint is
#           then portable: it can be read alone and compared with one produced by
#           an independent study. Cost: configurations occupying a narrow band of
#           the space look similar, because they are.
#   False — RELATIVE. Each axis is rescaled to its own cross-case maximum (padded
#           10%). Maximises visual separation within one comparison set, but the
#           figure is meaningless in isolation and cannot be overlaid on another
#           comparison set.
# Captions must match this setting; see CAPTION NOTE at the foot of this file.
ABSOLUTE_AXES = True

# ── Fingerprint axes ──
# Four axes, all in [0,1]; each is read by its own definition (paper Section 5):
# extent (WIN), shell concentration (STP), permitted action labels (SLT), adaptive performance (DDR).
#   Static layer  (functionals of the solved game: attractor + shield)
#     WIN  Winning Fraction   |W| / (|S_x| - |U|)
#     SEP  Shell Steepness    1 - H / ln(k_max)     (normalized entropy deficit)
#     SLT  Shield Latitude    sum|Safe| / sum|Act_D x V|   over shielded-reachable
#   Adaptive layer (functional of learned adaptive play)
#     DDR  Defender Dominance mean clean-host fraction
#
# AXIS ORDER IS DELIBERATE. On a 4-spoke radar the opposite pairs are (0,2) and
# (1,3). WIN sits opposite DDR (static state-mass vs. adaptive outcome) and
# SEP opposite SLT (the two static metrics that load on PC2 with opposite sign).
METRIC_LABELS = [
    "Winning\nFraction",
    "Shell\nSteepness",
    "Defender\nDominance",
    "Shield\nLatitude",
]
# Short axis codes (small multiples)
METRIC_CODES = ["WIN", "STP", "DDR", "SLT"]
METRIC_KEYS = [
    "winning_fraction",
    "shell_steepness",
    "defender_dominance",
    "shield_latitude",
]


def _get_radar_values(case_dict):
    """Extract radar values from a full case dict. Three come from safety_game (static layer); one is the mean
    defender dominance from the MARL results (adaptive layer). Every quantity
    is pulled straight from the JSON — no defaults invented. Shield latitude is
    computed by the analyzer over the shielded-reachable domain; state the
    domain in the caption."""
    sg = case_dict["safety_game"]
    vals = []
    for k in METRIC_KEYS:
        if k == "defender_dominance":
            v = case_dict["results"]["defender_dominance_pct"]["mean"] / 100.0
        else:
            v = sg[k]
        vals.append(v)
    return vals


# ═══════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════
def load_case(idx):
    path = os.path.join(RESULTS_DIR, f"bastion_{idx}_summary.json")
    with open(path, "r") as f:
        return json.load(f)


def load_all_cases():
    cases = {}
    for idx in CASE_INDICES:
        path = os.path.join(RESULTS_DIR, f"bastion_{idx}_summary.json")
        if os.path.exists(path):
            cases[idx] = load_case(idx)
        else:
            print(f"WARNING: {path} not found — skipping case {idx}")
    return cases


def load_case_episodes(idx):
    """Per-episode trajectories for one case, written by main.py as
    bastion_{idx}_episodes.npz. Arrays are (n_runs, n_episodes):
        clean_perc  per-episode mean clean-host fraction, in [0,1]
        reward_D    per-episode mean defender reward
        reward_A    per-episode mean attacker reward
    Only clean_perc is used by fig5. It is the quantity DDR averages, so the
    figure plots the integrand rather than a derived score."""
    path = os.path.join(RESULTS_DIR, f"bastion_{idx}_episodes.npz")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def load_all_episodes():
    """Same contract as load_all_cases: warn and skip rather than fail, so the
    other four figures still build if the .npz files are absent."""
    eps = {}
    for idx in CASE_INDICES:
        path = os.path.join(RESULTS_DIR, f"bastion_{idx}_episodes.npz")
        if os.path.exists(path):
            eps[idx] = load_case_episodes(idx)
        else:
            print(f"WARNING: {path} not found — skipping case {idx} in fig5")
    return eps


# ═══════════════════════════════════════════════════════════
#  STYLE SETUP
# ═══════════════════════════════════════════════════════════
def apply_academic_style():
    """Clean academic style — JAIR / NeurIPS / ICML ready."""
    plt.rcParams.update({
        "figure.facecolor":   BG_COLOR,
        "axes.facecolor":     BG_COLOR,
        "axes.edgecolor":     SPINE_COLOR,
        "axes.labelcolor":    TEXT_COLOR,
        "text.color":         TEXT_COLOR,
        "xtick.color":        TEXT_COLOR,
        "ytick.color":        TEXT_COLOR,
        "grid.color":         GRID_COLOR,
        "grid.alpha":         0.6,
        "legend.facecolor":   "#F5F5F5",
        "legend.edgecolor":   SPINE_COLOR,
        "legend.labelcolor":  TEXT_COLOR,
        "font.family":        "serif",
        "font.serif":         ["CMU Serif", "Computer Modern", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset":   "cm",
        "font.size":          10,
        "axes.titlesize":     12,
        "figure.titlesize":   14,
        "axes.linewidth":     0.8,
        "savefig.facecolor":  BG_COLOR,
        "savefig.edgecolor":  BG_COLOR,
        "savefig.dpi":        300,
    })


# ═══════════════════════════════════════════════════════════
#  FIGURE 1: SINGLE RADAR — BASELINE FINGERPRINT
# ═══════════════════════════════════════════════════════════
def fig1_baseline_fingerprint(cases):
    """Single radar chart for Case 1, introducing the visual vocabulary."""
    apply_academic_style()
    case = cases[1]

    values = _get_radar_values(case)
    maxima = _radar_maxima(cases)
    normed = [v / m if m > 0 else 0 for v, m in zip(values, maxima)]

    angles = np.linspace(0, 2 * np.pi, len(METRIC_KEYS), endpoint=False).tolist()
    normed_closed = normed + [normed[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Absolute mode: rings carry real values, every axis shares the [0,1] scale.
    # Relative mode: rings are unlabeled — each axis is on its own scale.
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    if ABSOLUTE_AXES:
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                           fontsize=7.5, color=MUTED_COLOR)
        ax.set_rlabel_position(22.5)      # between spokes, clear of the shape
        ax.set_ylim(0, 1.12)
    else:
        ax.set_yticklabels(["", "", "", ""])
        ax.set_ylim(0, 1.15)

    # Draw the shape
    ax.plot(angles_closed, normed_closed, color=COLORS[1], linewidth=2.5, zorder=3)
    ax.fill(angles_closed, normed_closed, color=COLORS[1], alpha=0.15, zorder=2)

    # Vertex dots with RAW value labels, offset outward so they survive downscaling
    for angle, norm_val, raw_val in zip(angles, normed, values):
        ax.scatter(angle, norm_val, color=COLORS[1], s=60, zorder=4,
                   edgecolors=TEXT_COLOR, linewidths=0.5)
        ax.text(angle, min(norm_val + 0.10, 1.08), f"{raw_val:.3f}",
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                color=TEXT_COLOR)

    # Axis labels
    ax.set_xticks(angles)
    ax.set_xticklabels(METRIC_LABELS, fontsize=9.5, fontweight="bold")
    ax.tick_params(axis="x", pad=10)

    # Spokes
    for angle in angles:
        ax.plot([angle, angle], [0, 1.0], color=SPINE_COLOR, linewidth=0.5, zorder=1)

    ax.spines["polar"].set_color(SPINE_COLOR)
    ax.grid(color=GRID_COLOR, alpha=0.5)

    if EMBED_TITLES:
        fig.suptitle("Defensibility Fingerprint — Baseline Topology",
                     fontsize=14, fontweight="bold", color=TEXT_COLOR, y=1.03)

    # Normalization disclosure — defuses the "why is it a balanced hexagon?" reviewer trap.
    # Each axis is scaled to its own cross-case maximum; printed numbers are raw scores.
    disclosure = "Metric values on a shared $[0,1]$ scale"
    ax.text(0.5, -0.12, disclosure,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color=MUTED_COLOR)

    _save_figure(fig, "fig1_baseline_fingerprint.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  FIGURE 2: FIVE FINGERPRINTS — SMALL MULTIPLES
# ═══════════════════════════════════════════════════════════
def fig2_five_fingerprints(cases):
    """5 radar charts as a 2x3 grid — readable at single-column width."""
    apply_academic_style()
    maxima = _radar_maxima(cases)
    angles = np.linspace(0, 2 * np.pi, len(METRIC_KEYS), endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor(BG_COLOR)

    for i, idx in enumerate(CASE_INDICES):
        if idx not in cases:
            continue
        case = cases[idx]
        values = _get_radar_values(case)
        normed = [v / m if m > 0 else 0 for v, m in zip(values, maxima)]
        normed_closed = normed + [normed[0]]

        ax = fig.add_subplot(2, 3, i + 1, polar=True)
        ax.set_facecolor(BG_COLOR)

        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        if ABSOLUTE_AXES:
            ax.set_yticklabels(["", "0.50", "", "1.00"], fontsize=7, color=MUTED_COLOR)
            ax.set_rlabel_position(22.5)
            ax.set_ylim(0, 1.08)
        else:
            ax.set_yticklabels([])
            ax.set_ylim(0, 1.15)

        color = COLORS[idx]
        ax.plot(angles_closed, normed_closed, color=color, linewidth=2.2, zorder=3)
        ax.fill(angles_closed, normed_closed, color=color, alpha=0.2, zorder=2)

        for angle, nv in zip(angles, normed):
            ax.scatter(angle, nv, color=color, s=42, zorder=4,
                       edgecolors=TEXT_COLOR, linewidths=0.4)

        ax.set_xticks(angles)
        ax.set_xticklabels(METRIC_CODES, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", pad=8)

        for angle in angles:
            ax.plot([angle, angle], [0, 1.0], color=SPINE_COLOR, linewidth=0.4, zorder=1)

        ax.spines["polar"].set_color(SPINE_COLOR)
        ax.grid(color=GRID_COLOR, alpha=0.4)

        # Per-panel title STAYS — it identifies the panel (caption can't do that).
        # Extra pad lifts it clear of the radar's top vertex labels (SEP/WIN).
        # Charcoal, not case-colored: the radar fill below already carries the color,
        # so colored text would be redundant and risks contrast/grayscale legibility.
        ax.set_title(f"Case {idx}: {CASE_DESCRIPTIONS[idx]}",
                     fontsize=12, fontweight="bold", color=TEXT_COLOR, pad=22)
        # |W| intentionally omitted here — it is fully reported in fig3 (decomposition),
        # and the fingerprint communicates shape, not magnitude.

    if EMBED_TITLES:
        fig.suptitle("Defensibility Fingerprints — What-If Comparison",
                     fontsize=16, fontweight="bold", color=TEXT_COLOR, y=0.99)

    # 6th cell -> axis-key legend (bold mono codes + dark roman names)
    ax_leg = fig.add_subplot(2, 3, 6)
    ax_leg.axis("off")
    ax_leg.text(0.05, 0.95, "Axis key", fontsize=14, fontweight="bold",
                color=TEXT_COLOR, va="top")
    key_pairs = [
        ("WIN", "Winning Fraction"),
        ("STP", "Shell Steepness"),
        ("DDR", "Defender Dominance"),
        ("SLT", "Shield Latitude"),
    ]
    y0, dy = 0.78, 0.115
    for r, (code, name) in enumerate(key_pairs):
        yy = y0 - r * dy
        ax_leg.text(0.05, yy, code, fontsize=12, fontweight="bold",
                    color=TEXT_COLOR, va="top", family="monospace")
        ax_leg.text(0.27, yy, name, fontsize=12, color=TEXT_COLOR, va="top")
    ax_leg.text(0.05, y0 - len(key_pairs) * dy - 0.06,
                "Metric values on a shared $[0,1]$ scale",
                fontsize=10, color=MUTED_COLOR, va="top")

    plt.tight_layout(rect=[0, 0.0, 1, 0.96 if EMBED_TITLES else 1.0])
    plt.subplots_adjust(hspace=0.62)
    _save_figure(fig, "fig2_five_fingerprints.png", bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  FIGURE 3: STATE SPACE DECOMPOSITION — STACKED BARS
# ═══════════════════════════════════════════════════════════
def fig3_state_space_decomposition(cases):
    """Horizontal stacked bars: Winning Region | Shell 3+ | Shell 2 | Shell 1
    (candidate states only)."""
    apply_academic_style()

    ZONE_COLORS = {
        "winning": "#1B998B",   # teal — safe ground
        "shell_1": "#C75146",   # muted red — 1 step from death
    }
    shell_color_map = ["#C75146", "#E8A838", "#7ECBC0", "#A8DDD5", "#C4EBE5"]

    fig, ax = plt.subplots(figsize=(14, 4.5))

    bar_height = 0.5
    y_positions = list(range(len(CASE_INDICES)))

    for i, idx in enumerate(CASE_INDICES):
        if idx not in cases:
            continue
        sg = cases[idx]["safety_game"]
        initial_U = sg["initial_unsafe_size"]
        candidates = sg["total_states"] - initial_U   # states that COULD have survived

        winning = sg["winning_region_size"]
        shells = sg.get("attractor_shells", [])        # [S1, S2, S3, ...]

        # Build segments: winning, then shells reversed (S3, S2, S1)
        segments = [winning]
        shells_reversed = list(reversed(shells))
        segments.extend(shells_reversed)

        seg_pct = [s / candidates * 100 for s in segments]

        n_shells = len(shells)
        shell_colors_reversed = []
        for j in range(n_shells):
            shell_colors_reversed.append(
                shell_color_map[min(n_shells - 1 - j, len(shell_color_map) - 1)]
            )
        colors = [ZONE_COLORS["winning"]] + shell_colors_reversed

        # Draw stacked horizontal bar
        left = 0
        for seg, col in zip(seg_pct, colors):
            ax.barh(i, seg, left=left, height=bar_height, color=col,
                    edgecolor="white", linewidth=0.8)
            if seg > 5:
                is_dark = col in [ZONE_COLORS["winning"], "#C75146"]
                ax.text(left + seg / 2, i, f"{seg:.1f}%",
                        ha="center", va="center", fontsize=13,
                        color="white" if is_dark else TEXT_COLOR)
            left += seg
        # |W| totals now live in the y-axis labels — keeps the x-axis honest at 100%.

    # Two-line y labels: case id + |W| total (pulled straight from JSON fields)
    y_labels = []
    for idx in CASE_INDICES:
        if idx not in cases:
            continue
        sg = cases[idx]["safety_game"]
        w, tot = sg["winning_region_size"], sg["total_states"]
        y_labels.append(f"Case {idx}")

    ax.set_yticks(y_positions[:len(y_labels)])
    ax.set_yticklabels(y_labels, fontsize=11)

    ax.set_xlabel("Candidate States (%)", fontsize=11, labelpad=8)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.invert_yaxis()

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    legend_elements = [
        Patch(facecolor=ZONE_COLORS["winning"], edgecolor="white", label="Winning Region"),
        Patch(facecolor="#7ECBC0",              edgecolor="white", label="Shell 3+"),
        Patch(facecolor="#E8A838",              edgecolor="white", label="Shell 2"),
        Patch(facecolor=ZONE_COLORS["shell_1"], edgecolor="white", label="Shell 1"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
              framealpha=0.9, ncol=4, handlelength=1.2, handletextpad=0.4, bbox_to_anchor=(1.0, 1.05),frameon=False)

    if EMBED_TITLES:
        fig.suptitle("Winning State Space Decomposition",
                     fontsize=13, fontweight="bold", color=TEXT_COLOR)

    plt.tight_layout()
    _save_figure(fig, "fig3_state_decomposition.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  FIGURE 4: DOMINANCE RATIO COMPARISON — GROUPED BOXPLOT
# ═══════════════════════════════════════════════════════════
def fig4_dominance_comparison(cases):
    """Side-by-side box plots of per-run L200 dominance ratios with CIs."""
    apply_academic_style()
    np.random.seed(RNG_SEED)   # reproducible jitter -> identical figure every rebuild

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    positions, labels = [], []

    for i, idx in enumerate(CASE_INDICES):
        if idx not in cases:
            continue
        dr = cases[idx]["results"]["defender_dominance_pct"]
        per_run = dr["per_run"]
        mean    = dr["mean"]
        ci_low  = dr["ci_low"]
        ci_high = dr["ci_high"]

        pos = i
        positions.append(pos)
        labels.append(f"Case {idx}")
        color = COLORS[idx]

        # Box plot
        ax.boxplot([per_run], positions=[pos], widths=0.45, patch_artist=True,
                   showfliers=False,
                   boxprops=dict(facecolor=color, alpha=0.25, edgecolor=color, linewidth=1.5),
                   medianprops=dict(color=TEXT_COLOR, linewidth=2),
                   whiskerprops=dict(color=color, linewidth=1.2),
                   capprops=dict(color=color, linewidth=1.2))

        # Individual run points (jittered)
        jitter = np.random.normal(0, 0.04, size=len(per_run))
        ax.scatter(np.full(len(per_run), pos) + jitter, per_run,
                   color=color, s=50, zorder=5, alpha=0.85,
                   edgecolors=TEXT_COLOR, linewidths=0.4)

        # 95% CI bar — widened + larger mean tick so it reads as deliberate, not an artifact
        ax.plot([pos + 0.30, pos + 0.30], [ci_low, ci_high],
                color=color, linewidth=4.5, solid_capstyle="round", alpha=0.75)
        ax.scatter([pos + 0.30], [mean], color=TEXT_COLOR, s=55, zorder=6,
                   marker="_", linewidths=2.2)

        # Mean annotation — lowered to sit just above the data. Charcoal, not case-colored:
        # the box below carries the color; colored text risks contrast/grayscale legibility.
        ax.text(pos, max(per_run) + 0.8, f"{mean:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=TEXT_COLOR)

    # 50% reference line — line stays light, label darkened + enlarged
    ax.axhline(50, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
    ax.text(len(positions) - 0.5, 50.6, "50% — Even Split",
            fontsize=9, color=MUTED_COLOR, ha="right")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
    ax.set_ylabel("Defender Dominance Ratio (L200) %", fontsize=11)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    # Case descriptions as secondary x-labels — upright, dark, larger
    # Case descriptions as secondary x-labels — upright, dark, larger.
    # Pushed to a 4.0-unit gap so they sit clear of the bold "Case N" tick labels.
    # To collapse to a SINGLE label instead, delete this loop and fold the description
    # into `labels` above (e.g. f"Case {idx}\n{CASE_DESCRIPTIONS[idx]}").
    ymin = ax.get_ylim()[0]
    for i, idx in enumerate(CASE_INDICES):
        if idx in cases:
            ax.text(i, ymin - 4.0, CASE_DESCRIPTIONS[idx],
                    ha="center", va="top", fontsize=10, color=TEXT_COLOR)

    if EMBED_TITLES:
        fig.suptitle("Defender Dominance at Equilibrium — What-If Comparison",
                     fontsize=13, fontweight="bold", color=TEXT_COLOR)

    plt.tight_layout()
    _save_figure(fig, "fig4_dominance_comparison.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  FIGURE 5: LEARNING TRAJECTORIES — CLEAN-HOST FRACTION
# ═══════════════════════════════════════════════════════════
# fig4 reports where adaptive play ENDS. fig5 reports how it GETS there, which
# the endpoint cannot recover: two configurations may converge to neighbouring
# operating points by opposite routes.
#
# The plotted quantity is the per-episode mean clean-host fraction — i.e. mu,
# the integrand DDR averages — not a derived score. The reader therefore watches
# the exact quantity being summarised, and the flat region confirms by
# inspection that the L200 window measures post-convergence behaviour rather
# than the tail of a transient.
#
# Only the seed MEAN is drawn. Per-seed dispersion is fig4's job; five std
# ribbons on one axis are unreadable, and drawing both invites the reader to
# compare ribbon widths that fig4 already states as confidence intervals.
#
# THE DETAIL PANEL HAS ONE JOB. Early training is dominated by exploration at
# epsilon = 0.5, so the first few hundred episodes are noisy for every case and
# a five-series zoom shows churn rather than structure. The panel therefore
# carries only the pair named in DETAIL_CASES, under heavier smoothing and with
# no raw trace, and exists to resolve a single claim: that the two exchange rank
# during training, having started on the opposite ordering to the one they
# finish on. Set DETAIL_CASES = None to show every case, or
# SHOW_DETAIL_PANEL = False to drop the panel and run the main axis full width.

L200_WINDOW   = 200    # shaded band; must match the L200 used in main.py
EARLY_WINDOW  = 600    # detail panel span — wide enough to contain the transient
SMOOTH_WINDOW = 25     # trailing moving average, in episodes; 1 disables smoothing
SHOW_RAW_TRACE = True  # faint unsmoothed curve behind each mean, so smoothing hides nothing
MARK_CROSSING  = True  # annotate where the two named cases exchange rank
CROSSING_PAIR  = (1, 3)  # baseline vs relaxed phi_A

SHOW_DETAIL_PANEL    = False
DETAIL_CASES         = (1, 3)  # None -> all cases; a tuple -> only those
DETAIL_SMOOTH_WINDOW = 60      # heavier than the main axis: this panel is a reading aid


def fig5_learning_trajectories(episodes):
    """Seed-mean clean-host fraction against episode.

    Left: every case over the full run. Right: the pair named in DETAIL_CASES
    over early training only.

    NO TEXT IS PLACED INSIDE EITHER AXIS. Reference lines and shaded regions are
    named in the legend; the single x-positioned callout sits above the top
    spine. In-axis labels cannot be positioned safely against data that changes
    between runs, and every placement rule eventually lands on a curve."""
    if not episodes:
        print("WARNING: no episode files loaded — skipping fig5")
        return

    apply_academic_style()

    show_detail = SHOW_DETAIL_PANEL and len(episodes) >= 1
    if show_detail:
        fig, (ax, axz) = plt.subplots(
            1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [3.4, 1.0]})
    else:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        axz = None
    fig.patch.set_facecolor(BG_COLOR)
    for a in (ax, axz):
        if a is not None:
            a.set_facecolor(BG_COLOR)

    n_ep = max(ep["clean_perc"].shape[1] for ep in episodes.values())
    early = min(EARLY_WINDOW, n_ep)
    detail_set = set(DETAIL_CASES) if DETAIL_CASES else set(CASE_INDICES)
    smoothed, handles = {}, []

    for idx in CASE_INDICES:
        if idx not in episodes:
            continue
        curve = episodes[idx]["clean_perc"].mean(axis=0) * 100.0
        smoothed[idx] = _rolling_mean(curve, SMOOTH_WINDOW)
        color = COLORS[idx]
        x = np.arange(len(curve))

        if SHOW_RAW_TRACE:
            ax.plot(x, curve, color=color, linewidth=0.6, alpha=0.15, zorder=2)
        ln, = ax.plot(x, smoothed[idx], color=color, linewidth=1.7, zorder=4,
                      label=f"Case {idx} — {CASE_DESCRIPTIONS[idx]}")
        handles.append(ln)

        # Detail panel: heavier smoothing, no raw trace. The raw signal is
        # already on the main axis; repeating it here would reintroduce exactly
        # the churn this panel exists to see past.
        if show_detail and idx in detail_set:
            axz.plot(x[:early], _rolling_mean(curve, DETAIL_SMOOTH_WINDOW)[:early],
                     color=color, linewidth=1.9, zorder=4)

    # Modest padding only. With no in-axis annotation there is nothing to
    # reserve headroom for, so the curves use the full panel.
    lo = min(c.min() for c in smoothed.values())
    hi = max(c.max() for c in smoothed.values())
    pad = 0.07 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(0, n_ep)

    # ── Shaded regions and reference line: named in the LEGEND ────────────
    # The L200 band is the region fig4 summarises; the point of showing it is
    # that every curve is already flat inside it.
    ax.axvspan(n_ep - L200_WINDOW, n_ep, color=MUTED_COLOR, alpha=0.10, zorder=1)
    if show_detail:
        ax.axvspan(0, early, color=ACCENT_COLOR, alpha=0.05, zorder=1)

    for a in (ax, axz):
        if a is not None:
            a.axhline(50, color="#999999", linestyle="--", linewidth=0.8,
                      alpha=0.7, zorder=1)

    handles.append(Line2D([0], [0], color="#999999", linestyle="--",
                          linewidth=0.8, label="50% — even split"))
    if show_detail:
        handles.append(Patch(facecolor=ACCENT_COLOR, alpha=0.05,
                             edgecolor="none",
                             label=f"detail range (episodes 0–{early})"))
    handles.append(Patch(facecolor=MUTED_COLOR, alpha=0.10, edgecolor="none",
                         label=f"L200 window (DDR measured here)"))

    # ── Rank exchange: marker inside, label ABOVE the top spine ───────────
    # Located from the smoothed curves, not eyeballed: the first episode after
    # which one case stays below the other for the remainder of training.
    if MARK_CROSSING and all(i in smoothed for i in CROSSING_PAIR):
        a_idx, b_idx = CROSSING_PAIR
        cross = _last_sign_change(smoothed[a_idx], smoothed[b_idx])
        if cross is not None:
            ax.axvline(cross, color=TEXT_COLOR, linestyle=":", linewidth=1.1,
                       alpha=0.85, zorder=3)
            # Align so a crossing near either end cannot overflow the panel.
            frac = cross / n_ep
            ha = "left" if frac < 0.12 else ("right" if frac > 0.88 else "center")
            ax.annotate(f"rank exchange — Case {b_idx} falls below Case {a_idx}, ep. {cross}",
                        xy=(cross, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, 6), textcoords="offset points",
                        ha=ha, va="bottom", fontsize=9, color=TEXT_COLOR,
                        annotation_clip=False)

    # ── Axes dressing ────────────────────────────────────────
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Clean-host fraction (%)", fontsize=11)

    if show_detail:
        axz.set_xlabel("Episode", fontsize=11)
        axz.set_xlim(0, early)
        shown = sorted(i for i in detail_set if i in smoothed)
        axz.set_title(f"Early training, episodes 0–{early}\n"
                      + " vs ".join(f"Case {i}" for i in shown),
                      fontsize=10.5, color=TEXT_COLOR)

    for a in (ax, axz):
        if a is None:
            continue
        for spine in ["top", "right"]:
            a.spines[spine].set_visible(False)
        a.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)
        a.set_axisbelow(True)

    # Legend below both panels. It carries the reference line and the shaded
    # regions as well as the cases, since none of them may be labelled in-axis.
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               framealpha=0.0, handlelength=1.8, columnspacing=2.0,
               bbox_to_anchor=(0.5, -0.015))

    if EMBED_TITLES:
        fig.suptitle("Learning Trajectories — Clean-Host Fraction",
                     fontsize=13, fontweight="bold", color=TEXT_COLOR)

    plt.tight_layout(rect=[0, 0.10, 1, 0.955])
    _save_figure(fig, "fig5_learning_trajectories.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  SUMMARY TABLE (prints to console + saves as text)
# ═══════════════════════════════════════════════════════════
def print_comparison_table(cases):
    """Print a formatted comparison table of all metrics across cases."""
    header = (f"{'Case':>6} | {'Description':<35} | {'WIN':>7} | {'STP':>7} | "
              f"{'SLT':>7} | {'|W|':>8} | {'W%':>6} | "
              f"{'DDR%':>8} | {'DDR CI':>18}")
    sep = "-" * len(header)
    print(f"\n{sep}")
    print("WHAT-IF COMPARISON — TOPOLOGY DEFENSIBILITY METRICS")
    print(sep)
    print(header)
    print(sep)

    rows = []
    for idx in CASE_INDICES:
        if idx not in cases:
            continue
        c = cases[idx]
        sg = c["safety_game"]
        dr = c["results"]["defender_dominance_pct"]

        row = (f"{idx:>6} | "
               f"{CASE_DESCRIPTIONS[idx]:<35} | "
               f"{sg['winning_fraction']:>7.3f} | "
               f"{sg['shell_steepness']:>7.3f} | "
               f"{sg['shield_latitude']:>7.3f} | "
               f"{sg['winning_region_size']:>8,} | "
               f"{sg['winning_region_pct']:>5.1f}% | "
               f"{dr['mean']:>7.1f}% | "
               f"[{dr['ci_low']:>6.1f}%, {dr['ci_high']:>6.1f}%]")
        rows.append(row)
        print(row)
    print(sep)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    table_path = os.path.join(OUTPUT_DIR, "comparison_table.txt")
    with open(table_path, "w") as f:
        f.write(f"{sep}\n")
        f.write("WHAT-IF COMPARISON — TOPOLOGY DEFENSIBILITY METRICS\n")
        f.write(f"{sep}\n{header}\n{sep}\n")
        f.write("\n".join(rows))
        f.write(f"\n{sep}\n")
    print(f"\nTable saved: {table_path}")


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════
def _radar_maxima(cases):
    """Per-axis scaling denominators.
    Absolute mode: 1.0 on every axis (all metrics are already in [0,1]).
    Relative mode: cross-case max per axis, padded 10% for breathing room.
    Pure function of the loaded data — no constants injected."""
    if ABSOLUTE_AXES:
        return [1.0] * len(METRIC_KEYS)
    maxima = [0.0] * len(METRIC_KEYS)
    for idx in CASE_INDICES:
        if idx not in cases:
            continue
        for j, v in enumerate(_get_radar_values(cases[idx])):
            maxima[j] = max(maxima[j], v)
    return [m * 1.1 if m > 0 else 1.0 for m in maxima]


def _rolling_mean(y, w):
    """Trailing moving average of width w, length preserved. The leading edge is
    padded with y[0] rather than using mode='same', which would taper the first
    w/2 points toward zero and manufacture a transient that is not in the data."""
    if w <= 1:
        return np.asarray(y, dtype=float)
    y = np.asarray(y, dtype=float)
    pad = np.concatenate([np.full(w - 1, y[0]), y])
    return np.convolve(pad, np.ones(w) / w, mode="valid")


def _last_sign_change(a, b):
    """First index after which a - b holds one sign for the remainder.
    Returns None if the two curves never exchange rank, so the annotation is
    simply omitted rather than pointing at a coincidental single-episode touch."""
    d = np.sign(np.asarray(a) - np.asarray(b))
    d[d == 0] = 1
    flips = np.flatnonzero(d[1:] != d[:-1])
    return int(flips[-1]) + 1 if len(flips) else None


def _save_figure(fig, filename, **kwargs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches=kwargs.get("bbox_inches", "tight"))
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
def main():
    cases = load_all_cases()
    if not cases:
        print("No case data found. Run the simulations first.")
        return

    print(f"Loaded {len(cases)} cases: {list(cases.keys())}")

    episodes = load_all_episodes()
    if episodes:
        print(f"Loaded episode traces for {len(episodes)} cases: {list(episodes.keys())}")

    print_comparison_table(cases)
    fig1_baseline_fingerprint(cases)
    fig2_five_fingerprints(cases)
    fig3_state_space_decomposition(cases)
    fig4_dominance_comparison(cases)
    fig5_learning_trajectories(episodes)

    print(f"\nAll paper figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

# ═══════════════════════════════════════════════════════════
#  CAPTION NOTE
# ═══════════════════════════════════════════════════════════
# The LaTeX captions for fig1 and fig2 must match ABSOLUTE_AXES.
#
# ABSOLUTE_AXES = True:
#   "The radius is absolute -- the rim is 1 on every axis and no cross-case
#    rescaling is applied -- so a fingerprint may be read on its own and
#    compared directly with one produced by an independent analysis."
#
# ABSOLUTE_AXES = False:
#   "Each axis is scaled to its maximum across the cases compared here;
#    printed values are raw metric scores. Shape differences across cases are
#    meaningful, but a single fingerprint is not interpretable in isolation and
#    cannot be overlaid on a fingerprint from a different comparison set."
#
# fig5 places NO TEXT INSIDE EITHER AXIS. The reference line and both shaded
# regions are named in the legend, and the rank-exchange callout sits above the
# top spine. Any future annotation must follow the same rule: in-axis labels
# cannot be positioned safely against data that changes between runs.
#
# fig5 has no scaling toggle, but three settings change what the figure asserts:
#
# SMOOTH_WINDOW > 1:
#   "Curves are seed means under a trailing moving average of SMOOTH_WINDOW
#    episodes; the unsmoothed seed mean is drawn faintly behind each curve."
#   Drop this sentence only if SMOOTH_WINDOW = 1.
#
# SHOW_RAW_TRACE = False:
#   remove the clause about the unsmoothed curve, since it is no longer drawn.
#
# DETAIL_CASES:
#   the right panel carries only these cases, under DETAIL_SMOOTH_WINDOW
#   smoothing and with no raw trace. Name them in the caption — the panel is
#   there to resolve one ordering claim, not to survey early training.
#
# The caption must also name the window: "the shaded band marks the final 200
# episodes over which DDR is measured." Without it a reader cannot tell that
# fig4 and fig5 report the same region of the same runs.
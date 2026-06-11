"""
What-If Comparison Script — Dual-Specification Safety Game Analysis
====================================================================
Loads all case_{IDX}_summary.json files from ../results/
and produces the 4 paper figures.

Requirements:
    Each summary.json must contain:
    - "results" → MARL metrics (reward, dominance, per_run, CIs)
    - "safety_game" → attractor/shield metrics from analyzer
    Case descriptions are defined in CASE_DESCRIPTIONS at the top of this file.

Run AFTER all 5 case studies have completed.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════
RESULTS_DIR = "../results"
OUTPUT_DIR = "../results/paper_figures"
CASE_INDICES = [1, 2, 3, 4, 5]

# ── Fill these in — displayed on figures and table ──
CASE_DESCRIPTIONS = {
    1: "Baseline topology",
    2: "Fully connected topology",
    3: "Relaxed φ_A: Unlimited destroys",
    4: "Relaxed φ_D: active >= 2",
    5: "VPN bypass removed",
}

# ── Academic palette: colorblind-safe, high contrast on white ──
COLORS = {
    1: "#2C5F8A",   # deep blue      — baseline (authoritative)
    2: "#D4652F",   # burnt orange    — full connectivity
    3: "#1B998B",   # teal           — relaxed φ_D
    4: "#A4243B",   # crimson        — relaxed φ_A
    5: "#5C4D7D",   # slate purple   — no VPN bypass
}
BG_COLOR     = "#FFFFFF"
GRID_COLOR   = "#E0E0E0"
TEXT_COLOR   = "#2D2D2D"
ACCENT_COLOR = "#2C5F8A"
SPINE_COLOR  = "#BDBDBD"

# Radar axis labels (short names for the chart)
METRIC_LABELS = [
    "Attackability",
    "Sinking\nRatio",
    "Shield\nFriction",
    "Attractor\nSteepness",
    "Violation\nProximity",
    "Attacker\nDominance",
]
METRIC_KEYS = [
    "attackability",
    "sinking_ratio",
    "shield_friction",
    "attractor_steepness",
    "mean_steps_to_violation",
    "attacker_dominance",
]
# All axes: higher = more dangerous. MSV is inverted (1/MSV) on radar only.
# Attacker dominance is derived (1 - DDR/100), not stored directly.
INVERT_ON_RADAR = {"mean_steps_to_violation"}


def _get_radar_values(case_dict):
    """Extract radar values from a full case dict. All axes: higher = worse.
    5 from safety_game, 1 derived from MARL results (attacker dominance ratio)."""
    sg = case_dict["safety_game"]
    vals = []
    for k in METRIC_KEYS:
        if k == "attacker_dominance":
            ddr = case_dict["results"]["defender_dominance_pct"]["mean"]
            v = 1.0 - (ddr / 100.0)
        else:
            v = sg[k]
            if k in INVERT_ON_RADAR:
                v = 1.0 / max(v - 1.0, 0.01)  # strip minimum, avoid div/0 at MSV=1.0
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


# ═══════════════════════════════════════════════════════════
#  STYLE SETUP
# ═══════════════════════════════════════════════════════════
def apply_academic_style():
    """Clean academic style — NeurIPS/ICML ready."""
    plt.rcParams.update({
        "figure.facecolor":   BG_COLOR,
        "axes.facecolor":     BG_COLOR,
        "axes.edgecolor":     SPINE_COLOR,
        "axes.labelcolor":    TEXT_COLOR,
        "text.color":         TEXT_COLOR,
        "xtick.color":       TEXT_COLOR,
        "ytick.color":       TEXT_COLOR,
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
    maxima = _compute_radar_maxima(cases)
    normed = [v / m if m > 0 else 0 for v, m in zip(values, maxima)]

    angles = np.linspace(0, 2 * np.pi, len(METRIC_KEYS), endpoint=False).tolist()
    normed_closed = normed + [normed[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Grid
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""], fontsize=6)
    ax.set_ylim(0, 1.15)
    # ax.set_ylim(0, 1.15)

    # Draw the shape
    ax.plot(angles_closed, normed_closed, color=COLORS[1], linewidth=2.5, zorder=3)
    ax.fill(angles_closed, normed_closed, color=COLORS[1], alpha=0.15, zorder=2)

    # Vertex dots with value labels
    for angle, norm_val, raw_val, label in zip(angles, normed, values, METRIC_LABELS):
        ax.scatter(angle, norm_val, color=COLORS[1], s=60, zorder=4, edgecolors=TEXT_COLOR, linewidths=0.5)
        # Raw value label just outside the point
        offset = 0.12
        ax.text(angle, norm_val + offset, f"{raw_val:.3f}",
                ha="center", va="center", fontsize=8, fontweight="bold",
                color=COLORS[1])

    # Axis labels
    ax.set_xticks(angles)
    ax.set_xticklabels(METRIC_LABELS, fontsize=9, fontweight="bold")

    # Spokes
    for angle in angles:
        ax.plot([angle, angle], [0, 1.0], color=SPINE_COLOR, linewidth=0.5, zorder=1)

    ax.spines["polar"].set_color(SPINE_COLOR)
    ax.grid(color=GRID_COLOR, alpha=0.5)

    fig.suptitle("Defensibility Fingerprint — Baseline Topology",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, y=0.98)
    ax.set_title(f"Case 1: {CASE_DESCRIPTIONS[1]}", fontsize=9, pad=20, color="#666666")

    _save_figure(fig, "fig1_baseline_fingerprint.png")
    plt.show()


# ═══════════════════════════════════════════════════════════
#  FIGURE 2: FIVE FINGERPRINTS — SMALL MULTIPLES
# ═══════════════════════════════════════════════════════════
def fig2_five_fingerprints(cases):
    """5 radar charts as a 2x3 grid — readable at single-column width."""
    apply_academic_style()
    maxima = _compute_radar_maxima(cases)
    angles = np.linspace(0, 2 * np.pi, len(METRIC_KEYS), endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    # 2 rows x 3 cols, near-square panels -> survives downscaling
    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor(BG_COLOR)

    short_labels = ["ATK", "SNK", "FRC", "STP", "VPX", "ADR"]

    for i, idx in enumerate(CASE_INDICES):
        if idx not in cases:
            continue
        case = cases[idx]
        sg = case["safety_game"]
        values = _get_radar_values(case)
        normed = [v / m if m > 0 else 0 for v, m in zip(values, maxima)]
        normed_closed = normed + [normed[0]]

        ax = fig.add_subplot(2, 3, i + 1, polar=True)   # <-- 2x3 grid
        ax.set_facecolor(BG_COLOR)

        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([])
        ax.set_ylim(0, 1.15)

        color = COLORS[idx]
        ax.plot(angles_closed, normed_closed, color=color, linewidth=2.2, zorder=3)
        ax.fill(angles_closed, normed_closed, color=color, alpha=0.2, zorder=2)

        for angle, nv in zip(angles, normed):
            ax.scatter(angle, nv, color=color, s=42, zorder=4,
                       edgecolors=TEXT_COLOR, linewidths=0.4)

        ax.set_xticks(angles)
        ax.set_xticklabels(short_labels, fontsize=12, fontweight="bold")  # <-- 7 -> 12
        ax.tick_params(axis='x', pad=8)                                   # breathing room

        for angle in angles:
            ax.plot([angle, angle], [0, 1.0], color=SPINE_COLOR, linewidth=0.4, zorder=1)

        ax.spines["polar"].set_color(SPINE_COLOR)
        ax.grid(color=GRID_COLOR, alpha=0.4)
        ax.set_title(f"Case {idx}: {CASE_DESCRIPTIONS[idx]}",
                     fontsize=12, fontweight="bold", color=color, pad=14)  # desc in title

        wr = sg.get("winning_region_size", "?")
        wr_pct = sg.get("winning_region_pct", "?")
        ax.text(0.5, -0.30, f"W = {wr} ({wr_pct}%)",
                transform=ax.transAxes, ha="center", fontsize=10, color="#666666")

    fig.suptitle("Defensibility Fingerprints — What-If Comparison",
                 fontsize=16, fontweight="bold", color=TEXT_COLOR, y=0.98)

    # 6th cell empty -> use it for the axis-key legend instead of a cramped footer
    ax_leg = fig.add_subplot(2, 3, 6)
    ax_leg.axis("off")
    key_lines = [
        "ATK  Attackability", "SNK  Sinking Ratio", "FRC  Shield Friction",
        "STP  Attractor Steepness", "VPX  Violation Proximity",
        "ADR  Attacker Dominance",
    ]
    ax_leg.text(0.05, 0.92, "Axis key", fontsize=13, fontweight="bold",
                color=TEXT_COLOR, va="top")
    ax_leg.text(0.05, 0.78, "\n".join(key_lines), fontsize=11,
                color="#444444", va="top", linespacing=1.6)

    plt.tight_layout(rect=[0, 0.0, 1, 0.94])
    plt.subplots_adjust(hspace=0.55)
    _save_figure(fig, "fig2_five_fingerprints.png", bbox_inches="tight")
    plt.show()


# ═══════════════════════════════════════════════════════════
#  FIGURE 3: STATE SPACE DECOMPOSITION — STACKED BARS
# ═══════════════════════════════════════════════════════════
def fig3_state_space_decomposition(cases):
    """Horizontal stacked bars: Winning Region | Shell 3 | Shell 2 | Shell 1 (candidate states only)."""
    apply_academic_style()
 
    # Sequential palette: safe (saturated) → danger (warm)
    ZONE_COLORS = {
        "winning":   "#1B998B",   # teal — safe ground
        "shell_1":   "#C75146",   # muted red — 1 step from death
    }
 
    fig, ax = plt.subplots(figsize=(14, 4.5))
 
    bar_height = 0.5
    y_positions = list(range(len(CASE_INDICES)))
 
    for i, idx in enumerate(CASE_INDICES):
        if idx not in cases:
            continue
        sg = cases[idx]["safety_game"]
        initial_U = sg["initial_unsafe_size"]
        candidates = sg["total_states"] - initial_U  # states that COULD have survived
 
        winning = sg["winning_region_size"]
        shells  = sg.get("attractor_shells", [])  # [S1, S2, S3, ...]
 
        # Build segments: winning, then shells reversed (S3, S2, S1)
        segments = [winning]
        shells_reversed = list(reversed(shells))
        segments.extend(shells_reversed)
 
        seg_pct = [s / candidates * 100 for s in segments]
 
        # Colors: winning, then shells from buffer to danger
        n_shells = len(shells)
        shell_color_map = ["#C75146", "#E8A838", "#7ECBC0", "#A8DDD5", "#C4EBE5"]
        shell_colors_reversed = []
        for j in range(n_shells):
            shell_colors_reversed.append(shell_color_map[min(n_shells - 1 - j, len(shell_color_map) - 1)])
 
        colors = [ZONE_COLORS["winning"]] + shell_colors_reversed
 
        # Draw stacked horizontal bar
        left = 0
        for seg, col in zip(seg_pct, colors):
            ax.barh(i, seg, left=left, height=bar_height, color=col,
                    edgecolor="white", linewidth=0.8)
            if seg > 5:
                is_dark = col in [ZONE_COLORS["winning"], "#C75146"]
                ax.text(left + seg / 2, i, f"{seg:.1f}%",
                        ha="center", va="center", fontsize=11,
                        color="white" if is_dark else TEXT_COLOR)
            left += seg
 
        # Annotations to the right
        ax.text(101, i, f"|W| = {winning:,}  ({winning/(candidates+initial_U)*100:.1f}%)",
                ha="left", va="center", fontsize=9, color=TEXT_COLOR)
 
    y_labels = [f"Case {idx}" for idx in CASE_INDICES if idx in cases]
    ax.set_yticks(y_positions[:len(y_labels)])
    ax.set_yticklabels(y_labels, fontsize=10)
 
    ax.set_xlabel("Candidate States (%)", fontsize=10, labelpad=8)
    ax.set_xlim(0, 118)
    ax.invert_yaxis()
 
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
 
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ZONE_COLORS["winning"], edgecolor="white", label="Winning Region"),
        Patch(facecolor="#7ECBC0",               edgecolor="white", label="Shell 3+"),
        Patch(facecolor="#E8A838",               edgecolor="white", label="Shell 2"),
        Patch(facecolor=ZONE_COLORS["shell_1"],  edgecolor="white", label="Shell 1"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.9, ncol=4, handlelength=1.2, handletextpad=0.4)
 
    fig.suptitle("Winning State Space Decomposition",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR)
 
    plt.tight_layout()
    _save_figure(fig, "fig3_state_decomposition.png")
    plt.show()


# ═══════════════════════════════════════════════════════════
#  FIGURE 4: DOMINANCE RATIO COMPARISON — GROUPED BOXPLOT
# ═══════════════════════════════════════════════════════════
def fig4_dominance_comparison(cases):
    """Side-by-side box plots of per-run L200 dominance ratios with CIs."""
    apply_academic_style()

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    positions = []
    labels = []
    all_data = []

    for i, idx in enumerate(CASE_INDICES):
        if idx not in cases:
            continue
        per_run = cases[idx]["results"]["defender_dominance_pct"]["per_run"]
        mean = cases[idx]["results"]["defender_dominance_pct"]["mean"]
        ci_low = cases[idx]["results"]["defender_dominance_pct"]["ci_low"]
        ci_high = cases[idx]["results"]["defender_dominance_pct"]["ci_high"]

        pos = i
        positions.append(pos)
        labels.append(f"Case {idx}")
        all_data.append(per_run)

        color = COLORS[idx]

        # Box plot
        bp = ax.boxplot([per_run], positions=[pos], widths=0.45, patch_artist=True,
                        showfliers=False,
                        boxprops=dict(facecolor=color, alpha=0.25, edgecolor=color, linewidth=1.5),
                        medianprops=dict(color=TEXT_COLOR, linewidth=2),
                        whiskerprops=dict(color=color, linewidth=1.2),
                        capprops=dict(color=color, linewidth=1.2))

        # Individual run points (jittered)
        jitter = np.random.normal(0, 0.04, size=len(per_run))
        ax.scatter([pos] * len(per_run) + jitter, per_run,
                   color=color, s=50, zorder=5, alpha=0.85,
                   edgecolors=TEXT_COLOR, linewidths=0.4)

        # CI bar
        ax.plot([pos + 0.28, pos + 0.28], [ci_low, ci_high],
                color=color, linewidth=3, solid_capstyle="round", alpha=0.7)
        ax.scatter([pos + 0.28], [mean], color=TEXT_COLOR, s=25, zorder=6, marker="_", linewidths=2)

        # Mean annotation
        ax.text(pos, max(per_run) + 1.5, f"{mean:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=color)

    # 50% reference line
    ax.axhline(50, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
    ax.text(len(positions) - 0.5, 50.5, "50% — Even Split", fontsize=7, color="#999999", ha="right")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylabel("Defender Dominance Ratio (L200) %", fontsize=10)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Light horizontal grid for readability
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    # Case descriptions as secondary x-labels
    for i, idx in enumerate(CASE_INDICES):
        if idx in cases:
            ax.text(i, ax.get_ylim()[0] - 2.5,
                    CASE_DESCRIPTIONS[idx],
                    ha="center", va="top", fontsize=7, color="#666666", style="italic")

    fig.suptitle("Defender Dominance at Equilibrium — What-If Comparison",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR)

    plt.tight_layout()
    _save_figure(fig, "fig4_dominance_comparison.png")
    plt.show()


# ═══════════════════════════════════════════════════════════
#  SUMMARY TABLE (prints to console + saves as text)
# ═══════════════════════════════════════════════════════════
def print_comparison_table(cases):
    """Print a formatted comparison table of all metrics across cases."""
    header = f"{'Case':>6} | {'Description':<35} | {'ATK':>7} | {'SNK':>7} | {'FRC':>7} | {'STP':>7} | {'MSV':>7} | {'|W|':>8} | {'W%':>6} | {'DDR%':>8} | {'DDR CI':>18}"
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
               f"{sg['attackability']:>7.3f} | "
               f"{sg['sinking_ratio']:>7.3f} | "
               f"{sg['shield_friction']:>7.3f} | "
               f"{sg['attractor_steepness']:>7.3f} | "
               f"{sg['mean_steps_to_violation']:>7.3f} | "
               f"{sg['winning_region_size']:>8,} | "
               f"{sg['winning_region_pct']:>5.1f}% | "
               f"{dr['mean']:>7.1f}% | "
               f"[{dr['ci_low']:>6.1f}%, {dr['ci_high']:>6.1f}%]")
        rows.append(row)
        print(row)
    print(sep)

    # Save to file
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
def _compute_radar_maxima(cases):
    """Compute per-axis max across all cases for consistent radar scaling."""
    maxima = [0.0] * len(METRIC_KEYS)
    for idx in CASE_INDICES:
        if idx not in cases:
            continue
        vals = _get_radar_values(cases[idx])
        for j, v in enumerate(vals):
            maxima[j] = max(maxima[j], v)
    # Ensure no zero maxima (avoid div/0), pad by 10% for visual breathing room
    return [m * 1.1 if m > 0 else 1.0 for m in maxima]


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

    print_comparison_table(cases)
    fig1_baseline_fingerprint(cases)
    fig2_five_fingerprints(cases)
    fig3_state_space_decomposition(cases)
    fig4_dominance_comparison(cases)

    print(f"\nAll paper figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
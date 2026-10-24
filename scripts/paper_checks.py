#!/usr/bin/env python3
"""
BASTION — supporting computations for statements in the paper
==============================================================

Outputs: paper_results/paper_checks.json and paper_results/paper_checks.log

  S5.3  share of absorbed non-initial mass in the first two shells (97.1-99.2%)
  S7.2  winning-region size difference between Cases 2 and 5 (548 positions)
  S7.3  the 128 positions whose attractor rank changes when the bypass edge is
        removed (Case 1 vs Case 5), and how many are reachable from v0 under
        admissible play
  S7.6  shield-latitude decomposition: Case 5 changes no permitted-action mask on
        retained shielded-reachable defender positions (80 positions leave the
        domain); Case 2 narrows menus at 56 positions
  S7.4, S7.5  Welch 95% intervals for DDR differences between configurations,
        computed from the ten per-run L200 means in results/bastion_N_summary.json

    python3 paper_checks.py
"""

import json, os
import numpy as np
from scipy import stats
import metric_suite as MS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "paper_results", "paper_checks.json")


def main():
    lines, R = [], {}
    def log(s=""): print(s); lines.append(s)
    G = {c: MS.load(c) for c in MS.CONFIGS}

    # S5.3 shell shares
    shares = {}
    for c, g in G.items():
        sh = np.diff(g["sizes"]); shares[c] = float((sh[0] + sh[1]) / sh.sum())
    log("S5.3  first-two-shell share of absorbed non-initial mass: " +
        ", ".join(f"case {c} {100*v:.1f}%" for c, v in shares.items()))
    R["first_two_shell_share"] = shares

    # S7.2 winning-region difference
    dW = int(G[5]["W"].sum() - G[2]["W"].sum())
    log(f"S7.2  |W| case 5 - case 2 = {dW}")
    R["W_diff_case5_case2"] = dW

    # S7.3 bypass rank changes and reachability from v0 under admissible play
    changed = G[1]["rank"] != G[5]["rank"]
    reach_adm = MS.reachable(G[1], shielded=False)
    reach_adm5 = MS.reachable(G[5], shielded=False)
    n_changed = int(changed.sum())
    n_reach = int((changed & (reach_adm | reach_adm5)).sum())
    log(f"S7.3  positions with changed rank (case 1 vs 5): {n_changed}; reachable from v0 under admissible play: {n_reach}")
    R["bypass_rank_changes"] = dict(changed=n_changed, reachable_from_v0=n_reach)

    # S7.6 shield-latitude decomposition over shielded-reachable defender positions
    def defender_domain(c):
        g = G[c]; Rs = MS.reachable(g, shielded=True)
        return set(int(v) for v in g["idxD"] if Rs[v] and g["W"][v])
    def mask(c, v): return G[c]["safeD"][G[c]["posD"][v]]
    D1 = defender_domain(1)
    dec = {}
    for c in (2, 5):
        Dc = defender_domain(c); common = D1 & Dc
        narrowed = sum(mask(c, v).sum() < mask(1, v).sum() for v in common)
        changed_m = sum((mask(c, v) != mask(1, v)).any() for v in common)
        dec[c] = dict(common=len(common), mask_changed=int(changed_m), narrowed=int(narrowed),
                      left_domain=len(D1 - Dc), entered_domain=len(Dc - D1))
        log(f"S7.6  case {c}: retained positions {len(common)}, mask changed {changed_m}, narrowed {narrowed}, "
            f"left domain {len(D1 - Dc)}, entered domain {len(Dc - D1)}")
    R["slt_decomposition"] = dec

    # S7.4 / S7.5 Welch intervals on per-run DDR (percentage points)
    per = {c: np.array(json.load(open(os.path.join(RESULTS, f"bastion_{c}_summary.json")))
                       ["results"]["defender_dominance_pct"]["per_run"]) for c in MS.CONFIGS}
    welch = {}
    for a, b in [(5, 2), (5, 4), (3, 2)]:
        x, y = per[a], per[b]; vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
        d = x.mean() - y.mean(); se = np.sqrt(vx + vy)
        df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
        t = stats.t.ppf(0.975, df)
        welch[f"{a}-{b}"] = dict(diff_pp=float(d), ci=[float(d - t * se), float(d + t * se)], df=float(df))
        log(f"S7.4/7.5  DDR case {a} - case {b} = {d:+.2f} pp, Welch 95% CI [{d - t*se:+.2f}, {d + t*se:+.2f}] (df {df:.1f})")
    R["welch_ddr_differences_pp"] = welch

    json.dump(R, open(OUT, "w"), indent=1)
    open(OUT.replace(".json", ".log"), "w").write("\n".join(lines) + "\n")
    print(f"written: {OUT} and {OUT.replace('.json', '.log')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
BASTION — Defensibility Metrics
===============================

Self-contained. Requires only numpy.

Vectorised reimplementation of the Bastion safety-game solver (~0.5 s per
150,000-state configuration), and computation of the four defensibility-
fingerprint metrics reported in the paper. All four are defender-oriented,
bounded in [0,1], larger = more defensible.

  Static layer (functionals of the solved game: attractor + shield)
    WIN  winning fraction   |W| / (|S_x| - |U|)
    STP  shell steepness    1 - H / ln(k_max)          normalized entropy deficit
    SLT  shield latitude    sum|Safe| / sum|Act_D x V| over the shielded-reachable domain
  Adaptive layer (functional of learned play, supplied from the MARL runs)
    DDR  defender dominance mean clean-host fraction

    python3 metric_suite.py            # values + independence + sensitivity
    python3 metric_suite.py --quick    # smaller corpora
    python3 metric_suite.py --only 1,3 # selected experiments

Experiments
-----------
  0  Solver validation (attractor sizes, |W| for the reference configuration)
  1  Metric values for the five paper configurations, with axis correlations
  2  Independence of the static metrics (random corpus: PCA, R^2, edge density)
  3  Two-layer sensitivity: topology vs specification perturbation

Results are printed and written to metric_suite_results.{json,log}.
"""

import argparse, itertools, json, math, os, time
import numpy as np

# ============================================================================
# ENCODING
# ============================================================================
# product index = h*48 + turn*24 + qD*4 + qA
#   h    in 0..3124   host status vector, base 5, host 0 most significant
#   turn 0 = attacker to move, 1 = defender to move
#   qD   in 0..5      itertools.product(["q0","q1","viol"], ["safe","viol"])
#   qA   in 0..3      q0, q1, q2, viol

C, X, D_, I, Z = 0, 1, 2, 3, 4
HN = ["C", "X", "D", "I", "Z"]
NH = 5
NHC = 5 ** NH
NPROD = NHC * 48
BAD = [X, D_, Z]          # "bad" for phi_D clause 1
DEAD = [I, Z]             # not operational for phi_D clause 2

TD = np.array([[C, C, I, C, C],          # defender: Noop Monitor Isolate Restore Fix
               [X, D_, I, X, X],
               [D_, D_, I, D_, D_],
               [I, I, I, C, I],
               [Z, Z, Z, Z, C]], dtype=np.int8)

TA = np.array([[C, X, C],                # attacker: Noop Spread Destroy
               [X, X, Z],
               [D_, D_, Z],
               [I, I, I],
               [Z, Z, Z]], dtype=np.int8)

QD = list(itertools.product(["q0", "q1", "viol"], ["safe", "viol"]))
QD_IDX = {q: i for i, q in enumerate(QD)}
D1 = {"q0": {0: "q0", 1: "q1", 2: "viol"},
      "q1": {0: "q0", 1: "q1", 2: "viol"},
      "viol": {0: "viol", 1: "viol", 2: "viol"}}
D2 = {"safe": {1: "safe", 0: "viol"}, "viol": {1: "viol", 0: "viol"}}
FD = np.array([("viol" not in q) for q in QD])
VIOL_A = 3
# (no discounted metric; nothing uses a gamma)

DTYPE = ["Noop", "Monitor", "Isolate", "Restore", "Fix"]
ATYPE = ["Noop", "Spread", "Destroy"]

_HV = np.zeros((NHC, NH), dtype=np.int8)
for _h in range(NHC):
    _r = _h
    for _j in range(NH - 1, -1, -1):
        _HV[_h, _j] = _r % 5
        _r //= 5
POW = np.array([5 ** (NH - 1 - j) for j in range(NH)], dtype=np.int64)


def decode(v):
    h, rem = v // 48, v % 48
    turn, qd, qa = rem // 24, (rem % 24) // 4, rem % 4
    hv = [HN[x] for x in _HV[h]]
    return (f"<{','.join(hv)}> turn={'A' if turn == 0 else 'D'} "
            f"qD={QD[qd]} qA={['q0','q1','q2','viol'][qa]}")


# ============================================================================
# SOLVER
# ============================================================================

def build_tables(topology, phid_active_min=3, attacker_counts_destroy=True):
    """Transition and automaton tables for one configuration."""
    top = np.array(topology, dtype=np.int8)

    dbbk = (np.isin(_HV[:, 3], BAD).astype(int) + np.isin(_HV[:, 4], BAD).astype(int))
    active = (~np.isin(_HV, DEAD)).sum(axis=1)
    sym1, sym2 = dbbk, (active >= phid_active_min).astype(int)

    qD_next = np.zeros((len(QD), NHC), dtype=np.int8)
    for qi, (a, b) in enumerate(QD):
        for h in range(NHC):
            qD_next[qi, h] = QD_IDX[(D1[a][int(sym1[h])], D2[b][int(sym2[h])])]

    nextH_D = np.zeros((NHC, 5 * NH), dtype=np.int64)
    for t in range(5):
        for host in range(NH):
            nh = _HV.copy()
            nh[:, host] = TD[_HV[:, host], t]
            nextH_D[:, t * NH + host] = nh.astype(np.int64) @ POW

    nextH_A = np.zeros((NHC, 3 * NH), dtype=np.int64)
    for t in range(3):
        for host in range(NH):
            nh = _HV.copy()
            if t == 1:                                   # Spread needs a neighbour
                nbrs = [u for u in range(NH) if u != host and top[host, u] == 1]
                spreadable = np.zeros(NHC, dtype=bool)
                for u in nbrs:
                    spreadable |= (_HV[:, u] == X) | (_HV[:, u] == D_)
                nh[:, host] = np.where(spreadable, TA[_HV[:, host], t], _HV[:, host])
            else:
                nh[:, host] = TA[_HV[:, host], t]
            nextH_A[:, t * NH + host] = nh.astype(np.int64) @ POW

    qA_next = np.zeros((4, 3 * NH), dtype=np.int8)
    for q in range(4):
        for t in range(3):
            for host in range(NH):
                a = t * NH + host
                qA_next[q, a] = (min(q + 1, VIOL_A)
                                 if (attacker_counts_destroy and t == 2) else q)
    return nextH_D, nextH_A, qD_next, qA_next


def solve(topology, phid_active_min=3, attacker_counts_destroy=True):
    """Synchronous (Jacobi) attractor. Returns game dict including rank."""
    nextH_D, nextH_A, qD_next, qA_next = build_tables(
        topology, phid_active_min, attacker_counts_destroy)

    hh = np.repeat(np.arange(NHC, dtype=np.int64), 48)
    rem = np.tile(np.arange(48, dtype=np.int64), NHC)
    turn, qd, qa = rem // 24, (rem % 24) // 4, rem % 4
    idxA, idxD = np.where(turn == 0)[0], np.where(turn == 1)[0]
    posA = -np.ones(NPROD, np.int64); posA[idxA] = np.arange(len(idxA))
    posD = -np.ones(NPROD, np.int64); posD[idxD] = np.arange(len(idxD))

    nH = nextH_A[hh[idxA]]
    nQD = qD_next[qd[idxA][:, None], nH]
    nQA = qA_next[qa[idxA][:, None], np.arange(15)[None, :]]
    succA = nH * 48 + 24 + nQD.astype(np.int64) * 4 + nQA.astype(np.int64)
    legalA = (nQA != VIOL_A)
    effA = (nH != hh[idxA][:, None])            # host vector actually changes

    nHd = nextH_D[hh[idxD]]
    nQDd = qD_next[qd[idxD][:, None], nHd]
    succD = nHd * 48 + nQDd.astype(np.int64) * 4 + qa[idxD][:, None]

    seed = ~FD[qd]
    in_a = seed.copy()
    rank = np.full(NPROD, -1, np.int32); rank[seed] = 0
    sizes = [int(seed.sum())]
    k = 0
    while True:
        k += 1
        new = in_a.copy()
        new[idxA[(in_a[succA] & legalA).any(axis=1)]] = True
        new[idxD[in_a[succD].all(axis=1)]] = True
        fresh = new & ~in_a
        if not fresh.any():
            break
        rank[fresh] = k
        in_a = new
        sizes.append(int(in_a.sum()))

    return dict(idxA=idxA, idxD=idxD, posA=posA, posD=posD,
                succA=succA, succD=succD, legalA=legalA, effA=effA,
                seed=seed, in_a=in_a, W=~in_a, rank=rank, sizes=sizes,
                qd=qd, qa=qa, hh=hh,
                safeD=~in_a[succD], safeA=(~in_a[succA]) & legalA)


S0 = (1 * 5 ** 4) * 48      # <X,C,C,C,C>, attacker to move, qD=(q0,safe), qA=q0


def reachable(G, shielded):
    """Vectorised BFS from s0. shielded=True confines both players to Safe(v)."""
    seen = np.zeros(NPROD, bool); seen[S0] = True
    frontier = np.array([S0], dtype=np.int64)
    while frontier.size:
        outs = []
        fa = frontier[G["posA"][frontier] >= 0]
        if fa.size:
            i = G["posA"][fa]
            m = G["safeA"][i] if shielded else G["legalA"][i]
            outs.append(G["succA"][i][m])
        fd = frontier[G["posD"][frontier] >= 0]
        if fd.size:
            j = G["posD"][fd]
            m = G["safeD"][j] if shielded else np.ones((len(j), 25), bool)
            outs.append(G["succD"][j][m])
        if not outs:
            break
        cand = np.unique(np.concatenate(outs))
        cand = cand[~seen[cand]]
        if cand.size == 0:
            break
        seen[cand] = True
        frontier = cand
    return seen


# ============================================================================
# METRICS  (defender-oriented, [0,1], larger = more defensible)
# ============================================================================

def shell_stats(sizes):
    A0, AK = sizes[0], sizes[-1]
    sh = np.array([sizes[i] - sizes[i - 1] for i in range(1, len(sizes))], float)
    K = len(sh)
    p = sh / sh.sum() if sh.sum() else sh
    H = -sum(x * math.log(x) for x in p if x > 0)
    return dict(A0=A0, AK=AK, K=K, H=H)


def WIN(G):
    """Winning fraction: |W| / (|S_x| - |U|). Of the non-violating states, the
    fraction from which the defender can guarantee safety."""
    A0 = G["sizes"][0]
    return float(G["W"].sum() / (NPROD - A0))


def STP(G):
    """Shell steepness: 1 - H / ln(k_max), normalized entropy deficit. Near 1 =
    attractor concentrated in the shallowest shells (mass hugs the violation
    boundary); near 0 = mass spread across many forced distances. Depends on the
    shape of the depth distribution, not the absorbed mass."""
    s = shell_stats(G["sizes"])
    return 1.0 if s["K"] <= 1 else float(1 - s["H"] / math.log(s["K"]))


def SLT(G, domain):
    """Shield latitude: mean fraction of defender actions the shield permits,
    over `domain` (required; the value depends strongly on the domain). Use the
    shielded-reachable set, the states a shielded play can actually enter."""
    sel = G["W"][G["idxD"]] & domain[G["idxD"]]
    if sel.sum() == 0:
        return float("nan")
    return float(G["safeD"][sel].sum() / (25 * sel.sum()))

# DDR (defender dominance) is measured by the MARL layer and supplied per
# configuration; it is not computed from the solved game.

# ============================================================================
# CONFIGURATIONS
# ============================================================================

BASE_TOP = [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0], [0, 0, 0, 1, 1]]

CONFIGS = {
    1: ("baseline",            dict(top=BASE_TOP)),
    2: ("fully connected",     dict(top=[[1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0],
                                         [1, 1, 1, 1, 0], [0, 0, 0, 1, 1]])),
    3: ("unlimited destroys",  dict(top=BASE_TOP, destroy=False)),
    4: ("phi_D active >= 2",   dict(top=BASE_TOP, amin=2)),
    5: ("no VPN bypass",       dict(top=[[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0, 1, 1, 0, 0],
                                         [0, 1, 1, 1, 0], [0, 0, 0, 1, 1]])),
}
# DDR read from the MARL runs (ten-seed L200 mean, as a fraction) in results/.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def _load_ddr():
    out = {}
    for c in CONFIGS:
        with open(os.path.join(RESULTS_DIR, f"bastion_{c}_summary.json")) as f:
            out[c] = json.load(f)["results"]["defender_dominance_pct"]["mean"] / 100.0
    return out


DDR = _load_ddr()


def load(c):
    _, cfg = CONFIGS[c]
    return solve(cfg["top"], cfg.get("amin", 3), cfg.get("destroy", True))

# ============================================================================
# STATISTICS
# ============================================================================

def _avg_rank(x):
    order = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1: r[m] = r[m].mean()
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    ra, rb = _avg_rank(a), _avg_rank(b)          # tie-corrected (average ranks)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def r2_on_others(M, key, others):
    y = M[key]
    X = np.column_stack([np.ones(len(y))] + [M[o] for o in others])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def pca(X, names, log):
    Z = (X - X.mean(0)) / X.std(0)
    w, V = np.linalg.eigh(np.cov(Z, rowvar=False))
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    ev = w / w.sum()
    log("  explained variance : " + "  ".join(f"PC{i+1}={x*100:5.1f}%" for i, x in enumerate(ev)))
    log("  loadings:")
    log("        " + "".join(f"{n:>9s}" for n in names))
    for i in range(len(names)):
        log(f"    PC{i+1} " + "".join(f"{V[j, i]:9.3f}" for j in range(len(names))))
    return ev.tolist(), V.tolist()


def make_corpus(n, mixed, seed, log):
    """Random configurations; returns WIN/STP/SLT per config plus edge density.
    mixed=False varies topology only; mixed=True also varies both specifications."""
    rng = np.random.default_rng(seed)
    rows, dens, tries = [], [], 0
    while len(rows) < n and tries < n * 6:
        tries += 1
        p = rng.uniform(0.15, 0.9)
        A = (rng.random((5, 5)) < p).astype(int); np.fill_diagonal(A, 1)
        amin = int(rng.choice([2, 3, 4])) if mixed else 3
        dest = bool(rng.random() > 0.35) if mixed else True
        try:
            G = solve(A.tolist(), amin, dest)
            if G["W"].sum() < 200 or len(G["sizes"]) < 2:
                continue
            Rs = reachable(G, True)
            slt = SLT(G, Rs)
            if not np.isfinite(slt):
                continue
            rows.append(dict(WIN=WIN(G), STP=STP(G), SLT=slt, v0_winning=bool(G["W"][S0])))
            dens.append((A.sum() - 5) / 20.0)
        except Exception:
            continue
    log(f"  corpus n = {len(rows)} ({'topology + specifications' if mixed else 'topology only'})")
    return rows, np.array(dens)

# ============================================================================
# EXPERIMENTS
# ============================================================================

def exp0(R, log):
    log("\n" + "=" * 74)
    log("EXPERIMENT 0 — solver validation (reference configuration)")
    log("=" * 74)
    t0 = time.time()
    G = load(1)
    log(f"  solve time              : {time.time()-t0:.2f}s")
    log(f"  attractor sizes         : {G['sizes']}")
    log(f"  |W|                     : {int(G['W'].sum())}   (expected 23730)")
    ok = int(G["W"].sum()) == 23730
    log(f"  VALIDATION              : {'PASS' if ok else 'FAIL'}")
    R["exp0"] = dict(sizes=G["sizes"], W=int(G["W"].sum()), passed=ok)


def exp1(R, log):
    log("\n" + "=" * 74)
    log("EXPERIMENT 1 — defensibility fingerprint, five configurations")
    log("=" * 74)
    log("  Defender-oriented, larger = more defensible. SLT over the shielded-")
    log("  reachable domain. DDR supplied from the MARL layer.")
    log(f"\n  {'case':22s}{'WIN':>8s}{'STP':>8s}{'SLT':>8s}{'DDR':>8s}{'|W|':>9s}")
    out = {}
    for c, (name, _) in CONFIGS.items():
        G = load(c)
        Rs = reachable(G, True)
        row = dict(WIN=WIN(G), STP=STP(G), SLT=SLT(G, Rs), DDR=DDR[c], W=int(G["W"].sum()))
        out[c] = row
        log(f"  {c}. {name:19s}{row['WIN']:8.4f}{row['STP']:8.4f}"
            f"{row['SLT']:8.4f}{row['DDR']:8.4f}{row['W']:9d}")
    K = ["WIN", "STP", "SLT", "DDR"]
    M = {k: np.array([out[c][k] for c in CONFIGS]) for k in K}
    Cm = np.corrcoef(np.vstack([M[k] for k in K]))
    log("\n  Pearson r among the four axes:")
    log("        " + "".join(f"{k:>9s}" for k in K))
    for i, k in enumerate(K):
        log(f"  {k:6s}" + "".join(f"{Cm[i, j]:9.3f}" for j in range(len(K))))
    mx = max(abs(Cm[3, j]) for j in range(3))
    log(f"\n  adaptive vs static: max |r| = {mx:.3f}  (n = {len(CONFIGS)} configurations)")
    R["exp1"] = dict(values=out, corr=Cm.tolist(), keys=K)


def exp2(R, log, n):
    log("\n" + "=" * 74)
    log("EXPERIMENT 2 — independence of the static metrics")
    log("=" * 74)
    rows, dens = make_corpus(n, mixed=True, seed=23, log=log)
    K = ["WIN", "STP", "SLT"]
    M = {k: np.array([r[k] for r in rows]) for k in K}
    log("\n  PCA on the three static metrics (mixed corpus):")
    ev, V = pca(np.array([[r[k] for k in K] for r in rows]), K, log)
    log("\n  variance explained by the other two (R^2):")
    for k in K:
        log(f"    {k:5s} R^2 = {r2_on_others(M, k, [o for o in K if o != k]):.4f}")
    rows_t, dens_t = make_corpus(n, mixed=False, seed=11, log=log)
    Mt = {k: np.array([r[k] for r in rows_t]) for k in K}
    log(f"\n  topology-only corpus: v0 winning in {sum(r['v0_winning'] for r in rows_t)} of {len(rows_t)} members")
    log("\n  Spearman (tie-corrected) vs edge density (topology-only corpus):")
    for k in K:
        log(f"    r({k:4s}, density) = {spearman(Mt[k], dens_t):+.3f}")
    R["exp2"] = dict(n=len(rows), explained=ev, loadings=V, keys=K,
                     r2={k: r2_on_others(M, k, [o for o in K if o != k]) for k in K},
                     density_corr={k: spearman(Mt[k], dens_t) for k in K},
                     topology_corpus_n=len(rows_t),
                     topology_corpus_v0_winning=int(sum(r["v0_winning"] for r in rows_t)))


def exp3(R, log):
    log("\n" + "=" * 74)
    log("EXPERIMENT 3 — two-layer sensitivity: topology vs specification")
    log("=" * 74)
    K = ["WIN", "STP", "SLT", "DDR"]
    vals = {}
    for c in CONFIGS:
        G = load(c)
        Rs = reachable(G, True)
        vals[c] = dict(WIN=WIN(G), STP=STP(G), SLT=SLT(G, Rs), DDR=DDR[c])
    b = {k: vals[1][k] for k in K}
    log(f"\n  percent change from baseline")
    log(f"  {'case':24s}" + "".join(f"{k:>9s}" for k in K))
    for c in (2, 5, 3, 4):
        log(f"  {c}. {CONFIGS[c][0]:21s}" +
            "".join(f"{100*(vals[c][k]-b[k])/abs(b[k]):+9.1f}" for k in K))
    topo = np.array([[abs(100 * (vals[c][k] - b[k]) / abs(b[k])) for k in K] for c in (2, 5)]).mean(0)
    spec = np.array([[abs(100 * (vals[c][k] - b[k]) / abs(b[k])) for k in K] for c in (3, 4)]).mean(0)
    log(f"\n  mean |%| topology     " + "".join(f"{x:9.2f}" for x in topo))
    log(f"  mean |%| specification" + "".join(f"{x:9.2f}" for x in spec))
    R["exp3"] = dict(values=vals, topo=dict(zip(K, topo.tolist())), spec=dict(zip(K, spec.tolist())))

# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--out", type=str, default="../paper_results/metric_suite_results.json")
    a = ap.parse_args()

    lines = []
    def log(s=""):
        print(s); lines.append(s)

    n = 25 if a.quick else 90
    sel = set(int(x) for x in a.only.split(",")) if a.only else {0, 1, 2, 3}

    R = {}
    t0 = time.time()
    log("BASTION defensibility metrics")
    log(f"numpy {np.__version__}")
    if 0 in sel: exp0(R, log)
    if 1 in sel: exp1(R, log)
    if 2 in sel: exp2(R, log, n)
    if 3 in sel: exp3(R, log)
    log(f"\ntotal runtime {time.time()-t0:.1f}s")

    with open(a.out, "w") as f:
        json.dump(R, f, indent=1, default=float)
    with open(a.out.replace(".json", ".log"), "w") as f:
        f.write("\n".join(lines))
    log(f"written: {a.out} and {a.out.replace('.json', '.log')}")


if __name__ == "__main__":
    main()
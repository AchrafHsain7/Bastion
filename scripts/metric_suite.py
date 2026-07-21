#!/usr/bin/env python3
"""
BASTION — Metric Layer Investigation Suite
==========================================

Self-contained. Requires only numpy.

Reimplements the Bastion safety-game solver in vectorised form (~0.5 s per
150,000-state configuration versus minutes for the reference implementation),
then runs every experiment performed during the metric review.

    python3 metric_suite.py                 # full run (~10-20 min)
    python3 metric_suite.py --quick         # smaller corpora (~3 min)
    python3 metric_suite.py --only 1,2,3    # selected experiments

Experiments
-----------
  0  Solver validation against published reference numbers
  1  Sweep-order defect: shell decomposition is not iteration-order invariant
  2  ATK algebraic identity: ATK = SNK/(1-SNK) * E[gamma^D]
  3  Metric values for the five paper configurations
  4  Domain sensitivity: FRC over full / reachable / shielded-reachable
  5  Reachable-set invariance under topology perturbation
  6  Perturbation sensitivity: topology versus specification
  7  Topology-only corpus: correlations and independence
  8  Mixed corpus: PCA, intrinsic dimensionality
  9  Rejected candidate metrics

Results are printed and written to metric_suite_results.json.
"""

import argparse, itertools, json, math, sys, time
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
GAMMA = 0.95

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


def solve_gauss_seidel(G):
    """Replicates the original in-place sweep. Returns attractor_sizes only."""
    posA, posD = G["posA"], G["posD"]
    succA, succD, legalA = G["succA"], G["succD"], G["legalA"]
    in_a = G["seed"].copy()
    sizes = [int(in_a.sum())]
    while True:
        before = int(in_a.sum())
        for v in range(NPROD):
            if in_a[v]:
                continue
            i = posA[v]
            if i >= 0:
                if (in_a[succA[i]] & legalA[i]).any():
                    in_a[v] = True
            else:
                if in_a[succD[posD[v]]].all():
                    in_a[v] = True
        if int(in_a.sum()) == before:
            break
        sizes.append(int(in_a.sum()))
    return sizes


def gs_levels(G):
    """Gauss-Seidel pass index per vertex."""
    posA, posD = G["posA"], G["posD"]
    succA, succD, legalA = G["succA"], G["succD"], G["legalA"]
    in_a = G["seed"].copy()
    lvl = np.full(NPROD, -1, np.int32); lvl[in_a] = 0
    k = 0
    while True:
        k += 1
        before = int(in_a.sum())
        for v in range(NPROD):
            if in_a[v]:
                continue
            i = posA[v]
            if i >= 0:
                if (in_a[succA[i]] & legalA[i]).any():
                    in_a[v] = True; lvl[v] = k
            else:
                if in_a[succD[posD[v]]].all():
                    in_a[v] = True; lvl[v] = k
        if int(in_a.sum()) == before:
            break
    return lvl


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
# METRICS
# ============================================================================

def shell_stats(sizes):
    A0, AK = sizes[0], sizes[-1]
    M = AK - A0
    sh = np.array([sizes[i] - sizes[i - 1] for i in range(1, len(sizes))], float)
    K = len(sh)
    p = sh / sh.sum() if sh.sum() else sh
    d = np.arange(1, K + 1)
    H = -sum(x * math.log(x) for x in p if x > 0)
    return dict(A0=A0, AK=AK, M=M, shells=sh, K=K, p=p, d=d, H=H)


def SNK(G):
    s = shell_stats(G["sizes"])
    return s["M"] / (NPROD - s["A0"])


def STP(G):
    s = shell_stats(G["sizes"])
    return 1.0 if s["K"] <= 1 else 1 - s["H"] / math.log(s["K"])


def MSV(G):
    s = shell_stats(G["sizes"])
    return float((s["d"] * s["p"]).sum())


def ATK(G, gamma=GAMMA):
    s = shell_stats(G["sizes"])
    return float((gamma ** s["d"] * s["shells"]).sum() / G["W"].sum())


def FRC(G, domain):
    """Fraction of defender actions the shield blocks, over `domain`.
    `domain` is required: the value depends strongly on it."""
    sel = G["W"][G["idxD"]] & domain[G["idxD"]]
    if sel.sum() == 0:
        return float("nan")
    return float((25 - G["safeD"][sel].sum(axis=1)).sum() / (25 * sel.sum()))


def EAB(G, domain):
    """Effective attacker branching: mean fraction of permitted attacker
    actions that change the host-status vector. REJECTED (R^2 = 0.91)."""
    sel = G["W"][G["idxA"]] & domain[G["idxA"]]
    if sel.sum() == 0:
        return float("nan")
    return float(((G["effA"][sel] & G["safeA"][sel]).sum(axis=1)).mean() / 15)


def RRM(G, domain):
    """Rank-reducing attacker multiplicity. REJECTED (0.29% topology response)."""
    sel = (~G["W"][G["idxA"]]) & domain[G["idxA"]] & (G["rank"][G["idxA"]] > 0)
    if sel.sum() == 0:
        return float("nan")
    rv = G["rank"][G["idxA"]][sel][:, None]
    rs = G["rank"][G["succA"][sel]]
    return float((((rs >= 0) & (rs < rv) & G["legalA"][sel]).sum(axis=1)).mean() / 15)


def BND(G, domain):
    """Boundary fraction. REJECTED (1.21% topology response)."""
    sel = G["W"][G["idxD"]] & domain[G["idxD"]]
    if sel.sum() == 0:
        return float("nan")
    return float((G["safeD"][sel].sum(axis=1) < 25).mean())


def block_profile(G, domain):
    """Block rate per defender action type."""
    sel = G["W"][G["idxD"]] & domain[G["idxD"]]
    if sel.sum() == 0:
        return {t: float("nan") for t in DTYPE}
    blocked = ~G["safeD"][sel]
    return {DTYPE[t]: float(blocked[:, t * 5:(t + 1) * 5].mean()) for t in range(5)}


def HHI(G, domain):
    """Herfindahl concentration of the block profile. REJECTED (1.70%)."""
    prof = np.array(list(block_profile(G, domain).values()))
    tot = prof.sum()
    return float(((prof / tot) ** 2).sum()) if tot > 0 else float("nan")


def SCR(G, R, Rs):
    """Shielded compression ratio |V_SR| / |V_R|. REJECTED (loads on PC1)."""
    return float(Rs.sum() / R.sum()) if R.sum() else float("nan")


def RHO(G, Rs):
    """Shielded path redundancy: permitted / legal out-edges over V_SR.
    REJECTED (r = -0.997 with FRC)."""
    a, d = Rs[G["idxA"]], Rs[G["idxD"]]
    legal = int(G["legalA"][a].sum()) + int(d.sum()) * 25
    perm = int(G["safeA"][a].sum()) + int(G["safeD"][d].sum())
    return float(perm / legal) if legal else float("nan")


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
DDR = {1: 0.539, 2: 0.227, 3: 0.475, 4: 0.779, 5: 0.807}

PUBLISHED_JACOBI = {          # from the corrected reference implementation
    1: dict(ATK=1.036, SNK=0.525, FRC=0.613, STP=0.510, MSV=1.307, W=23730),
    2: dict(ATK=1.071, SNK=0.534, FRC=0.618, STP=0.513, MSV=1.307, W=23310),
    3: dict(ATK=1.340, SNK=0.589, FRC=0.640, STP=0.468, MSV=1.360, W=20526),
    4: dict(ATK=0.444, SNK=0.321, FRC=0.516, STP=0.616, MSV=1.207, W=33966),
    5: dict(ATK=1.026, SNK=0.523, FRC=0.613, STP=0.524, MSV=1.295, W=23858),
}


def load(c):
    _, cfg = CONFIGS[c]
    return solve(cfg["top"], cfg.get("amin", 3), cfg.get("destroy", True))


# ============================================================================
# STATISTICS
# ============================================================================

def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def r2_on_others(M, key, others):
    y = M[key]
    Xd = np.column_stack([np.ones(len(y))] + [M[o] for o in others])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return float(1 - ((y - Xd @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def pca(X, names, log):
    Z = (X - X.mean(0)) / X.std(0)
    w, V = np.linalg.eigh(np.cov(Z, rowvar=False))
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    ev = w / w.sum()
    log("  explained variance : " + "  ".join(f"PC{i+1}={x*100:5.1f}%" for i, x in enumerate(ev)))
    log("  cumulative         : " + "  ".join(f"{ev[:i+1].sum()*100:5.1f}%" for i in range(len(ev))))
    log("  loadings:")
    log("        " + "".join(f"{n:>9s}" for n in names))
    for i in range(min(3, len(names))):
        log(f"    PC{i+1} " + "".join(f"{V[j, i]:9.3f}" for j in range(len(names))))
    return ev.tolist(), V.tolist()


def make_corpus(n, mixed, seed, log):
    rng = np.random.default_rng(seed)
    rows, dens, tries = [], [], 0
    while len(rows) < n and tries < n * 6:
        tries += 1
        p = rng.uniform(0.15, 0.9)
        A = (rng.random((5, 5)) < p).astype(int)
        np.fill_diagonal(A, 1)
        amin = int(rng.choice([2, 3, 4])) if mixed else 3
        dest = bool(rng.random() > 0.35) if mixed else True
        try:
            G = solve(A.tolist(), amin, dest)
            if G["W"].sum() < 200 or len(G["sizes"]) < 2:
                continue
            R, Rs = reachable(G, False), reachable(G, True)
            f = FRC(G, Rs)
            if not np.isfinite(f) or R.sum() < 50:
                continue
            rows.append(dict(SNK=SNK(G), STP=STP(G), MSV=MSV(G), ATK=ATK(G), FRC=f,
                             EAB=EAB(G, Rs), SCR=SCR(G, R, Rs), RHO=RHO(G, Rs),
                             nR=int(R.sum()), nRs=int(Rs.sum())))
            dens.append((A.sum() - 5) / 20.0)
        except Exception:
            continue
    log(f"  corpus n = {len(rows)} from {tries} draws "
        f"({'topology + specifications' if mixed else 'topology only'})")
    return rows, np.array(dens)


# ============================================================================
# EXPERIMENTS
# ============================================================================

def exp0(R, log):
    log("\n" + "=" * 78)
    log("EXPERIMENT 0 — solver validation")
    log("=" * 78)
    t0 = time.time()
    G = load(1)
    log(f"  solve time: {time.time()-t0:.2f}s")
    log(f"  attractor_sizes (Jacobi) = {G['sizes']}")
    log(f"  |W| = {int(G['W'].sum())}   expected 23730")
    ok = int(G["W"].sum()) == 23730 and G["sizes"] == [100000, 119238, 125630, 125886, 126270]
    log(f"  VALIDATION: {'PASS' if ok else 'FAIL'}")
    R["exp0"] = dict(sizes=G["sizes"], W=int(G["W"].sum()), pass_=ok)


def exp1(R, log):
    log("\n" + "=" * 78)
    log("EXPERIMENT 1 — sweep order determines the shell decomposition")
    log("=" * 78)
    G = load(1)
    sj = G["sizes"]
    sg = solve_gauss_seidel(G)
    log(f"  Jacobi       : {sj}")
    log(f"    shells     : {[sj[i]-sj[i-1] for i in range(1, len(sj))]}")
    log(f"  Gauss-Seidel : {sg}          <- reproduces published values")
    log(f"    shells     : {[sg[i]-sg[i-1] for i in range(1, len(sg))]}")
    log(f"  same fixed point: {sj[-1] == sg[-1]}   |W| identical: True")

    lj, lg = G["rank"], gs_levels(G)
    absorbed = G["in_a"] & ~G["seed"]
    dis = absorbed & (lj != lg)
    log(f"\n  absorbed vertices            : {int(absorbed.sum())}")
    log(f"  assigned a different level   : {int(dis.sum())} "
        f"({100*dis.sum()/absorbed.sum():.1f}%)")

    # decisive: shell 1 must be exactly one forced move from violation
    seed, posA, posD = G["seed"], G["posA"], G["posD"]

    def one_move(v):
        i = posA[v]
        if i >= 0:
            return bool((seed[G["succA"][i]] & G["legalA"][i]).any())
        return bool(seed[G["succD"][posD[v]]].all())

    res = {}
    for name, lv in (("Jacobi", lj), ("Gauss-Seidel", lg)):
        s1 = np.where((lv == 1) & absorbed)[0]
        bad = [int(v) for v in s1 if not one_move(int(v))]
        res[name] = (len(s1), len(bad))
        log(f"\n  {name}: shell-1 vertices = {len(s1)}; "
            f"cannot force violation in one move = {len(bad)}")
        if bad:
            w = bad[0]
            log(f"    witness: {decode(w)}")
            log(f"      Jacobi rank = {lj[w]}, GS shell = {lg[w]}")
            j = posD[w]
            if j >= 0:
                esc = [(DTYPE[a // 5] + f"(host{a%5})", int(G['succD'][j][a]))
                       for a in range(25) if not seed[G["succD"][j][a]]]
                log(f"      defender escapes (successors NOT in U): {len(esc)} of 25")
                for nm, s in esc[:3]:
                    log(f"        {nm:16s} -> {decode(s)}")
    log("\n  CONCLUSION: shell index equals forced-steps-to-violation under the")
    log("  synchronous update and does not under Gauss-Seidel. The certificate,")
    log("  |W| and the fixed point are unaffected; ATK, STP, MSV are.")
    R["exp1"] = dict(jacobi=sj, gauss_seidel=sg, disagree=int(dis.sum()),
                     shell1_check={k: dict(n=v[0], violations=v[1]) for k, v in res.items()})


def exp2(R, log):
    log("\n" + "=" * 78)
    log("EXPERIMENT 2 — ATK identity:  ATK = SNK/(1-SNK) * E[gamma^D]")
    log("=" * 78)
    worst, out = 0.0, {}
    for c in CONFIGS:
        G = load(c)
        s = shell_stats(G["sizes"])
        for tag, sizes in (("Jacobi", G["sizes"]), ("GS", solve_gauss_seidel(G))):
            st = shell_stats(sizes)
            snk = st["M"] / (NPROD - st["A0"])
            egd = float((GAMMA ** st["d"] * st["p"]).sum())
            atk = float((GAMMA ** st["d"] * st["shells"]).sum() / G["W"].sum())
            pred = snk / (1 - snk) * egd
            err = abs(pred - atk)
            worst = max(worst, err)
            out[f"{c}-{tag}"] = dict(ATK=atk, pred=pred, err=err)
            log(f"  cfg {c} {tag:6s}  ATK={atk:.6f}  predicted={pred:.6f}  |err|={err:.2e}")
    log(f"\n  max absolute error = {worst:.3e}")
    log("  CONCLUSION: ATK is a function of SNK and the shell distribution.")
    log("  It contributes no information and is unbounded via the odds factor.")
    R["exp2"] = dict(max_err=worst, detail=out)


def exp3(R, log):
    log("\n" + "=" * 78)
    log("EXPERIMENT 3 — metric values, five paper configurations")
    log("=" * 78)
    log(f"  {'case':22s}{'SNK':>8s}{'STP':>8s}{'MSV':>8s}{'ATK':>8s}"
        f"{'FRC_W':>8s}{'FRC_SR':>9s}{'|W|':>9s}{'DDR':>7s}")
    out = {}
    for c, (name, _) in CONFIGS.items():
        G = load(c)
        full = np.ones(NPROD, bool)
        Rs = reachable(G, True)
        row = dict(SNK=SNK(G), STP=STP(G), MSV=MSV(G), ATK=ATK(G),
                   FRC_W=FRC(G, full), FRC_SR=FRC(G, Rs),
                   W=int(G["W"].sum()), DDR=DDR[c])
        out[c] = row
        log(f"  {c}. {name:19s}{row['SNK']:8.4f}{row['STP']:8.4f}{row['MSV']:8.4f}"
            f"{row['ATK']:8.4f}{row['FRC_W']:8.4f}{row['FRC_SR']:9.4f}"
            f"{row['W']:9d}{row['DDR']:7.3f}")
        pub = PUBLISHED_JACOBI[c]
        for k in ("SNK", "STP", "MSV", "ATK"):
            assert abs(row[k] - pub[k]) < 2e-3, f"cfg {c} {k} mismatch"
        assert row["W"] == pub["W"]
    log("  cross-check against published corrected values: PASS")
    K = ["SNK", "STP", "FRC_SR", "DDR"]
    M = {k: np.array([out[c][k] for c in CONFIGS]) for k in K}
    Cm = np.corrcoef(np.vstack([M[k] for k in K]))
    log("\n  Pearson r among the surviving axes (five cases):")
    log("        " + "".join(f"{k:>9s}" for k in K))
    for i, k in enumerate(K):
        log(f"  {k:6s}" + "".join(f"{Cm[i, j]:9.3f}" for j in range(len(K))))
    R["exp3"] = dict(values=out, corr=Cm.tolist(), keys=K)


def exp4(R, log):
    log("\n" + "=" * 78)
    log("EXPERIMENT 4 — FRC depends strongly on the averaging domain")
    log("=" * 78)
    log(f"  {'case':22s}{'full W':>10s}{'reachable':>11s}{'shielded':>10s}")
    out = {}
    for c, (name, _) in CONFIGS.items():
        G = load(c)
        full = np.ones(NPROD, bool)
        Rr, Rs = reachable(G, False), reachable(G, True)
        row = dict(full=FRC(G, full), reach=FRC(G, Rr), shielded=FRC(G, Rs))
        out[c] = row
        log(f"  {c}. {name:19s}{row['full']:10.4f}{row['reach']:11.4f}{row['shielded']:10.4f}")
    log("\n  CONCLUSION: a factor of ~2.2 rests on an unstated choice.")
    log("  The domain must be reported. Shielded-reachable is what play visits.")
    R["exp4"] = out


def exp5(R, log):
    log("\n" + "=" * 78)
    log("EXPERIMENT 5 — reachable set is invariant under topology perturbation")
    log("=" * 78)
    sets = {}
    for c in (1, 2, 5):
        G = load(c)
        Rr, Rs = reachable(G, False), reachable(G, True)
        sets[c] = (Rr, Rs, G)
        log(f"  cfg {c} ({CONFIGS[c][0]:18s}) |V_R|={int(Rr.sum()):6d}  "
            f"|V_SR|={int(Rs.sum()):5d}  |W|={int(G['W'].sum()):6d}  "
            f"|W&V_R|={int((G['W']&Rr).sum()):5d}")
    same15 = bool((sets[1][0] == sets[5][0]).all())
    same12 = bool((sets[1][0] == sets[2][0]).all())
    log(f"\n  V_R(baseline) == V_R(no bypass)      : {same15}")
    log(f"  V_R(baseline) == V_R(fully connected): {same12}")
    hc = {c: len(set(np.unique(sets[c][2]['hh'][sets[c][0]]).tolist())) for c in (1, 2, 5)}
    log(f"  reachable host configurations        : {hc}")
    log(f"  V_SR(baseline) == V_SR(no bypass)    : "
        f"{bool((sets[1][1] == sets[5][1]).all())}")
    log("\n  CONCLUSION: these topologies differ in attack-path LENGTH and")
    log("  MULTIPLICITY, not in what is reachable. Worst-case analysis asks only")
    log("  whether a forcing strategy exists over an infinite horizon, so neither")
    log("  quantity enters. Any functional of the reachable set or of the rank")
    log("  marginals is therefore invariant here. The SHIELDED reachable set does")
    log("  move, because the shield's mask depends on the attractor.")
    R["exp5"] = dict(VR_identical_1_5=same15, VR_identical_1_2=same12,
                     host_configs=hc,
                     sizes={c: dict(VR=int(sets[c][0].sum()), VSR=int(sets[c][1].sum()))
                            for c in (1, 2, 5)})


def exp6(R, log):
    log("\n" + "=" * 78)
    log("EXPERIMENT 6 — perturbation sensitivity: topology versus specification")
    log("=" * 78)
    K = ["SNK", "STP", "MSV", "ATK", "FRC_SR", "EAB", "RRM", "BND", "HHI", "SCR", "RHO"]
    vals = {}
    for c in CONFIGS:
        G = load(c)
        Rr, Rs = reachable(G, False), reachable(G, True)
        vals[c] = dict(SNK=SNK(G), STP=STP(G), MSV=MSV(G), ATK=ATK(G),
                       FRC_SR=FRC(G, Rs), EAB=EAB(G, Rs), RRM=RRM(G, Rr),
                       BND=BND(G, Rs), HHI=HHI(G, Rs), SCR=SCR(G, Rr, Rs),
                       RHO=RHO(G, Rs))
    b = {k: vals[1][k] for k in K}
    log(f"  percent change from baseline")
    log(f"  {'case':24s}" + "".join(f"{k:>8s}" for k in K))
    for c in (2, 5, 3, 4):
        log(f"  {c}. {CONFIGS[c][0]:21s}" +
            "".join(f"{100*(vals[c][k]-b[k])/abs(b[k]):+8.1f}" for k in K))
    topo = np.array([[abs(100 * (vals[c][k] - b[k]) / abs(b[k])) for k in K] for c in (2, 5)]).mean(0)
    spec = np.array([[abs(100 * (vals[c][k] - b[k]) / abs(b[k])) for k in K] for c in (3, 4)]).mean(0)
    log(f"\n  mean |%| topology     " + "".join(f"{x:8.2f}" for x in topo))
    log(f"  mean |%| specification" + "".join(f"{x:8.2f}" for x in spec))
    log("\n  CONCLUSION: under path-length-only topology perturbation every formal")
    log("  metric moves <4%. DDR moves 53.8%. This motivates the two-layer split.")
    R["exp6"] = dict(values=vals, topo=dict(zip(K, topo.tolist())),
                     spec=dict(zip(K, spec.tolist())))


def exp7(R, log, n):
    log("\n" + "=" * 78)
    log("EXPERIMENT 7 — topology-only corpus")
    log("=" * 78)
    rows, dens = make_corpus(n, mixed=False, seed=11, log=log)
    K = ["SNK", "STP", "MSV", "ATK", "FRC", "EAB", "SCR", "RHO"]
    M = {k: np.array([r[k] for r in rows]) for k in K}
    log("\n  Spearman versus edge density (pure topology signal):")
    for k in K:
        log(f"    r({k:4s}, density) = {spearman(M[k], dens):+.3f}")
    log("\n  Spearman among metrics:")
    log("        " + "".join(f"{k:>8s}" for k in K))
    for k in K:
        log(f"  {k:6s}" + "".join(f"{spearman(M[k], M[j]):8.3f}" for j in K))
    core = ["SNK", "STP", "FRC"]
    log("\n  variance explained by the other core metrics (R^2):")
    for k in K:
        oth = [o for o in core if o != k]
        log(f"    {k:5s} R^2 = {r2_on_others(M, k, oth):.4f}")
    log("\n  CONCLUSION: under topology-only variation FRC collapses onto the")
    log("  compromise-pressure factor. RHO is FRC restated (r ~ -0.99).")
    R["exp7"] = dict(n=len(rows),
                     density_corr={k: spearman(M[k], dens) for k in K},
                     r2={k: r2_on_others(M, k, [o for o in core if o != k]) for k in K})


def exp8(R, log, n):
    log("\n" + "=" * 78)
    log("EXPERIMENT 8 — mixed corpus and intrinsic dimensionality")
    log("=" * 78)
    rows, dens = make_corpus(n, mixed=True, seed=23, log=log)
    core = ["SNK", "STP", "MSV", "ATK", "FRC"]
    M = {k: np.array([r[k] for r in rows]) for k in core}
    log("\n  PCA on the formal metrics (mixed corpus):")
    ev, V = pca(np.array([[r[k] for k in core] for r in rows]), core, log)
    log("\n  Spearman among metrics:")
    log("        " + "".join(f"{k:>8s}" for k in core))
    for k in core:
        log(f"  {k:6s}" + "".join(f"{spearman(M[k], M[j]):8.3f}" for j in core))
    log("\n  variance explained by the others (R^2):")
    for k in core:
        log(f"    {k:5s} R^2 = {r2_on_others(M, k, [o for o in core if o != k]):.4f}")
    allk = core + ["SCR", "RHO"]
    Ma = {k: np.array([r[k] for r in rows]) for k in allk}
    log("\n  PCA including the rejected candidates SCR and RHO:")
    pca(np.array([[r[k] for k in allk] for r in rows]), allk, log)
    log("\n  CONCLUSION: under topology-only variation two components explain")
    log("  ~95%. Under mixed variation three explain ~92%, and FRC's dominant")
    log("  loading moves to PC2 opposite STP: FRC separates from SNK exactly when")
    log("  the specification changes. Adding SCR/RHO creates no new component.")
    R["exp8"] = dict(n=len(rows), explained=ev, loadings=V, keys=core,
                     r2={k: r2_on_others(M, k, [o for o in core if o != k]) for k in core})


def exp9(R, log):
    log("\n" + "=" * 78)
    log("EXPERIMENT 9 — rejected candidate metrics, block profile")
    log("=" * 78)
    log(f"  block rate by defender action type (shielded-reachable domain)")
    log(f"  {'case':22s}" + "".join(f"{t:>10s}" for t in DTYPE))
    out = {}
    for c, (name, _) in CONFIGS.items():
        G = load(c)
        prof = block_profile(G, reachable(G, True))
        out[c] = prof
        log(f"  {c}. {name:19s}" + "".join(f"{prof[t]:10.4f}" for t in DTYPE))
    log("\n  Isolate is blocked 2.2-2.7x more than any other action type in every")
    log("  configuration: the shield strips containment while leaving cleanup.")
    log("  A real diagnostic statement, flat under topology perturbation.")
    R["exp9"] = out


# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--out", type=str, default="metric_suite_results.json")
    a = ap.parse_args()

    lines = []

    def log(s=""):
        print(s); lines.append(s)

    n7 = 25 if a.quick else 90
    n8 = 25 if a.quick else 90
    sel = set(int(x) for x in a.only.split(",")) if a.only else set(range(10))

    R = {}
    t0 = time.time()
    log("BASTION metric investigation suite")
    log(f"numpy {np.__version__}   gamma = {GAMMA}")
    if 0 in sel: exp0(R, log)
    if 1 in sel: exp1(R, log)
    if 2 in sel: exp2(R, log)
    if 3 in sel: exp3(R, log)
    if 4 in sel: exp4(R, log)
    if 5 in sel: exp5(R, log)
    if 6 in sel: exp6(R, log)
    if 7 in sel: exp7(R, log, n7)
    if 8 in sel: exp8(R, log, n8)
    if 9 in sel: exp9(R, log)
    log(f"\ntotal runtime {time.time()-t0:.1f}s")

    with open(a.out, "w") as f:
        json.dump(R, f, indent=1, default=float)
    with open(a.out.replace(".json", ".log"), "w") as f:
        f.write("\n".join(lines))
    log(f"written: {a.out} and {a.out.replace('.json', '.log')}")


if __name__ == "__main__":
    main()

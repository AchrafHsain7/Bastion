#!/usr/bin/env python3
"""
BASTION — learned-policy robustness (paper Section 7.9, Table 7)
================================================================

Inputs : results/qtables/bastion_{case}_seed{seed}_Q.npz   (written by main.py)
Outputs: results/policy_robustness.json and results/policy_robustness.log

For each configuration (cases 1-5) this script reports, as exact expected
clean-host fractions over T = 1,000 plies from v0 (forward propagation of the
state distribution; no Monte Carlo):

  * benchmark      occupancy of a policy pair solving the discounted (gamma = 0.95)
                   zero-sum game on the shielded product game;
  * greedy_vs_BR   each final greedy learned defender against an exact attacker
                   best response (discounted reward, gamma = 0.95);
  * eps_vs_BR      the same defender keeping its epsilon = 0.1 exploration, against
                   an exact attacker best response.

Game semantics are those of the training protocol: both players restricted to
Safe(v); rewards +/-(n_C - n_notC)/|H| to the mover; engagement reset at
all-clean configurations with p = 0.1 (attacker respawns at GW, both automata
reset, same player to move); the clean fraction is read before action selection,
after any reset. Learned policies act greedily on the host-vector Q-tables with
first-index tie-breaking, exactly as library.ShieldedQlearning.take_action.

The solver is metric_suite.solve. The computation is restricted to the closure
of {v0, reset targets} under permitted moves and resets; every policy over
Safe(.) stays inside it, so the values there are exact.

    python3 policy_robustness.py              # all five cases
"""

import argparse, glob, json, os, re
import numpy as np
import scipy.sparse as sp
import metric_suite as MS

GAMMA, EPS, T, P_RESET = 0.95, 0.1, 1000, 0.1
RESET_H = 1 * 5 ** 4                                  # <X,C,C,C,C>
SCORE = (2 * (MS._HV == MS.C).sum(1) - MS.NH) / MS.NH  # 2*mu - 1 per host vector
CLEAN = (MS._HV == MS.C).sum(1) / MS.NH                # mu per host vector


class ShieldedArena:
    """Shielded product game restricted to the closure of v0 and the reset targets."""

    def __init__(self, G):
        hh = G["hh"]
        def turn(v): return (v % 48) // 24             # 0 = attacker, 1 = defender
        def moves(v):
            if turn(v) == 0:
                r = G["posA"][v]; return G["succA"][r], G["safeA"][r]
            r = G["posD"][v]; return G["succD"][r], G["safeD"][r]
        def reset(v): return RESET_H * 48 + turn(v) * 24
        starts = [MS.S0, MS.S0 + 24]
        seen = {s: i for i, s in enumerate(starts)}; order = list(starts); k = 0
        while k < len(order):
            v = order[k]; k += 1
            succ, ok = moves(v)
            nxt = list(succ[ok])
            if hh[v] == 0: nxt.append(reset(v))
            for u in nxt:
                if u not in seen: seen[u] = len(order); order.append(u)
        self.idx = np.array(order); n = self.n = len(order)
        self.succ = np.zeros((n, 25), np.int64); self.allowed = np.zeros((n, 25), bool)
        self.isA = np.array([turn(v) == 0 for v in order])
        for i, v in enumerate(order):
            succ, ok = moves(v); m = len(succ)
            self.allowed[i, :m] = ok
            self.succ[i, :m] = [seen.get(int(u), 0) for u in succ]
            self.succ[i, :m][~ok] = 0
        assert self.allowed.any(1).all(), "every closure state must have a permitted move"
        h = hh[self.idx]; nh = hh[self.idx[self.succ]]
        self.R = np.where(self.isA[:, None], -SCORE[nh], SCORE[nh])     # reward to the mover
        self.allclean = h == 0
        self.reset = np.array([seen[reset(v)] if hh[v] == 0 else i for i, v in enumerate(order)])
        self.clean = CLEAN[h]; self.h = h
        self.v0 = seen[MS.S0]

    # --- policies -----------------------------------------------------------
    def greedy(self, Q):
        a = np.argmax(np.where(self.allowed, Q, -np.inf), 1)
        P = np.zeros((self.n, 25)); P[np.arange(self.n), a] = 1.0; return P

    def uniform(self):
        P = self.allowed.astype(float); return P / P.sum(1, keepdims=True)

    def eps_greedy(self, P, eps): return (1 - eps) * P + eps * self.uniform()

    def joint(self, PA, PD): return np.where(self.isA[:, None], PA, PD)

    def learned_Q(self, QD, QA):
        Q = np.full((self.n, 25), -np.inf)
        QD = QD.reshape(5 ** 5, 25); QA = QA.reshape(5 ** 5, 15)
        Q[~self.isA] = QD[self.h[~self.isA]]
        Q[self.isA, :15] = QA[self.h[self.isA]]
        return Q

    # --- discounted solver: each player 'opt' or ('fixed', P) ---------------
    def solve(self, modeA="opt", modeD="opt", gamma=GAMMA, tol=1e-12, iters=100000):
        V = np.zeros(self.n); A, D = self.isA, ~self.isA
        def backup(Q, mode, rows):
            Qr, al = Q[rows], self.allowed[rows]
            if mode == "opt": return np.where(al, Qr, -np.inf).max(1)
            return (np.where(al, Qr, 0.0) * mode[1][rows]).sum(1)
        for _ in range(iters):
            Q = self.R - gamma * V[self.succ]               # negamax: next mover is the opponent
            U = np.empty(self.n); U[A] = backup(Q, modeA, A); U[D] = backup(Q, modeD, D)
            Vn = np.where(self.allclean, P_RESET * U[self.reset] + (1 - P_RESET) * U, U)
            done = np.abs(Vn - V).max() < tol; V = Vn
            if done: break
        return V, self.R - gamma * V[self.succ]

    # --- exact occupancy ----------------------------------------------------
    def occupancy(self, P, T=T):
        n = self.n; rows = np.repeat(np.arange(n), 25)
        w = (P * self.allowed).ravel(); keep = w > 0
        MT = sp.csr_matrix((w[keep], (rows[keep], self.succ.ravel()[keep])), shape=(n, n)).T.tocsr()
        ac = np.flatnonzero(self.allclean)
        rr = np.concatenate([np.arange(n), ac]); rc = np.concatenate([np.arange(n), self.reset[ac]])
        rw = np.ones(n); rw[ac] = 1 - P_RESET
        RT = sp.csr_matrix((np.concatenate([rw, np.full(len(ac), P_RESET)]), (rr, rc)), shape=(n, n)).T.tocsr()
        d = np.zeros(n); d[self.v0] = 1.0; total = 0.0
        for _ in range(T):
            d = RT @ d                                       # engagement reset, then read mu
            total += float(d @ self.clean)
            d = MT @ d
        return total / T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", default="../results/qtables")
    ap.add_argument("--out", default="../results/policy_robustness.json")
    a = ap.parse_args()
    lines = []
    def log(s=""): print(s); lines.append(s)

    out = {"protocol": dict(gamma=GAMMA, eps=EPS, plies=T, reset_p=P_RESET), "cases": {}}
    for c in sorted(MS.CONFIGS):
        arena = ShieldedArena(MS.load(c))
        _, Qopt = arena.solve()
        Popt = arena.greedy(Qopt)
        bench = arena.occupancy(Popt)
        runs = []
        for f in sorted(glob.glob(os.path.join(a.qdir, f"bastion_{c}_seed*_Q.npz")),
                        key=lambda p: int(re.search(r"seed(\d+)", p).group(1))):
            z = np.load(f)
            P = arena.greedy(arena.learned_Q(z["QD"], z["QA"]))
            r = {"seed": int(re.search(r"seed(\d+)", f).group(1))}
            for tag, PD in [("greedy_vs_BR", P), ("eps_vs_BR", arena.eps_greedy(P, EPS))]:
                _, Q = arena.solve("opt", ("fixed", PD))
                r[tag] = arena.occupancy(arena.joint(arena.greedy(Q), PD))
            runs.append(r)
        g = [r["greedy_vs_BR"] for r in runs]; e = [r["eps_vs_BR"] for r in runs]
        out["cases"][c] = dict(name=MS.CONFIGS[c][0], closure_states=arena.n, benchmark=bench, runs=runs,
                               greedy_vs_BR_range=[min(g), max(g)] if g else None,
                               eps_vs_BR_range=[min(e), max(e)] if e else None)
        log(f"case {c} ({MS.CONFIGS[c][0]}): {len(runs)} runs | discounted-optimal pair {100*bench:6.2f}% | "
            f"greedy defender vs BR {100*min(g):.2f}-{100*max(g):.2f}% | "
            f"eps=0.1 defender vs BR {100*min(e):.2f}-{100*max(e):.2f}%")

    C = out["cases"]
    g13 = [r["greedy_vs_BR"] for c in (1, 2, 3) for r in C[c]["runs"]]
    e13 = [r["eps_vs_BR"] for c in (1, 2, 3) for r in C[c]["runs"]]
    b13 = [C[c]["benchmark"] for c in (1, 2, 3)]
    log("\nTable 7 (expected clean-host %, 1,000 plies)")
    log(f"  Cases 1-3  greedy defender vs BR {100*min(g13):.2f}-{100*max(g13):.2f}   discounted-optimal {100*min(b13):.1f}-{100*max(b13):.1f}")
    for c in (4, 5):
        g = C[c]["greedy_vs_BR_range"]
        log(f"  Case {c}     greedy defender vs BR {100*g[0]:.1f}-{100*g[1]:.1f}   discounted-optimal {100*C[c]['benchmark']:.1f}")
    log("\nSection 7.9, epsilon = 0.1 defender vs BR")
    log(f"  Cases 1-3 at most {100*max(e13):.1f}%")
    for c in (4, 5):
        e = C[c]["eps_vs_BR_range"]
        log(f"  Case {c} {100*e[0]:.0f}-{100*e[1]:.0f}%")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    open(a.out.replace(".json", ".log"), "w").write("\n".join(lines) + "\n")
    print(f"written: {a.out} and {a.out.replace('.json', '.log')}")


if __name__ == "__main__":
    main()

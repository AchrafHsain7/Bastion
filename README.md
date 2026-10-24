## License
- **Code** (`scripts/`): MIT — see [LICENSE](LICENSE).
- **Data & experimental outputs** (`results/`, `paper_results/`): CC-BY-4.0 — see [LICENSE-DATA](LICENSE-DATA).

# Reproducing the paper

Requirements: Python 3.12, `numpy`, `scipy`, `matplotlib`, `seaborn`, `tqdm`. Run all commands from `scripts/`.

| Paper content | Command | Output |
|---|---|---|
| Adaptive layer: DDR, Table 5 DDR column and intervals, Figures 6–7 (10 seeds × 3,000 episodes per case) | `python3 main.py N` for N = 1…5 | `results/bastion_N_summary.json`, `results/bastion_N_episodes.npz`, `results/qtables/bastion_N_seedS_Q.npz` |
| Static layer: Tables 5–6, Section 7.4/7.6 percentage changes, Discussion corpus statistics | `python3 metric_suite.py` | `paper_results/metric_suite_results.{json,log}` |
| Section 7.9 / Table 7: learned-policy robustness and the discounted-optimal benchmark | `python3 policy_robustness.py` | `results/policy_robustness.{json,log}` |
| Sections 5.3, 7.2, 7.3, 7.6 counts; Welch intervals in Sections 7.4–7.5 | `python3 paper_checks.py` | `paper_results/paper_checks.{json,log}` |
| Figures 4, 5, 6, 8 and the comparison table | `python3 whatif_visuals.py` | `paper_results/` |

`main.py N` reproduces the stored `results/` arrays bit-for-bit (seeds 1–10). Saving the Q-tables consumes no random numbers.

# Experimental outputs

Each case N ∈ {1..5} corresponds to a configuration in the paper's Table 3.

- `results/bastion_N_episodes.npz` — raw per-seed, per-episode arrays (10 seeds × 3,000 episodes). Keys:
  - `reward_D`, `reward_A` — mean defender and attacker reward per episode;
  - `clean_perc` — mean clean-host fraction per episode.
- `results/bastion_N_summary.json` — aggregated DDR, 95% CI (Student's t), and the safety-game readings.
- `results/bastion_N_*.png` — learning curves, box plots, convergence distributions.
- `results/qtables/bastion_N_seedS_Q.npz` — final learned Q-tables of each run. Keys:
  - `QD` (5,5,5,5,5,25) and `QA` (5,5,5,5,5,15), indexed by host-status vector (C, X, D, I, Z = 0…4; GW, Web, WS, DB, BK);
  - `epsilon_final`.
- `results/policy_robustness.json` — per case and per run:
  - `benchmark` — occupancy of a discounted-optimal (γ = 0.95) policy pair;
  - `greedy_vs_BR` — each final greedy defender against an exact attacker best response;
  - `eps_vs_BR` — the same defender with ε = 0.1 exploration, against an exact attacker best response.

  All values are exact expected clean-host fractions over 1,000 plies from v0, under the training protocol's shield and engagement resets.
- `paper_results/metric_suite_results.{json,log}` — static metrics, perturbation percentages and topology-corpus statistics. DDR is read from `results/`.
- `paper_results/paper_checks.{json,log}` — shell shares, the bypass rank-change reachability check, the shield-latitude decomposition, and Welch intervals for DDR differences.

## License
- **Code** (`scripts/`): MIT — see [LICENSE](LICENSE).
- **Data & experimental outputs** (`results/`): CC-BY-4.0 — see [LICENSE-DATA](LICENSE-DATA).

# Experimental outputs
Each case N ∈ {1..5} corresponds to a what-if configuration (see paper, Table of cases).
- `bastion_N_episodes.npz` — raw per-seed, per-episode arrays (10 seeds × 3,000 episodes). Keys: <list array names>.
- `bastion_N_summary.json` — aggregated DDR, 95% CI (Student's t), winning-region size.
- `bastion_N_*.png` — learning curves, box plots, convergence distributions.
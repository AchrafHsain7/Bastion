#!/usr/bin/env bash
# Full reproduction of the paper's results. Run from anywhere: ./scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
export MPLBACKEND=Agg OMP_NUM_THREADS=1   # no plot windows; avoid thread oversubscription
mkdir -p ../logs

# 1. Training: the five cases in parallel (independent processes, identical results)
pids=()
for N in 1 2 3 4 5; do
    python3 main.py "$N" > "../logs/main_$N.log" 2>&1 &
    pids+=($!)
done
fail=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "main.py $((i+1)) FAILED, see logs/main_$((i+1)).log"; fail=1; }
done
[ "$fail" -eq 0 ] || exit 1
echo "training done"

# 2. Analyses (each reads the training outputs; seconds each)
python3 metric_suite.py      > ../logs/metric_suite.log 2>&1
python3 policy_robustness.py > ../logs/policy_robustness.log 2>&1
python3 paper_checks.py      > ../logs/paper_checks.log 2>&1
python3 whatif_visuals.py    > ../logs/whatif_visuals.log 2>&1
echo "all done; outputs in results/ and paper_results/, logs in logs/"      
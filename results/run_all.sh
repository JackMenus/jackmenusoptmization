#!/bin/sh
# Reproduces every experiment recorded in results/. Run from the repository root.
set -x
optiforge benchmark configs/benchmark_portfolio.yaml > results/benchmark_portfolio.log 2>&1
for c in rastrigin pressure_vessel knapsack_exact milp pd_control facility_mixed; do
  optiforge solve configs/solve_$c.yaml > results/solve_$c.log 2>&1
done
for c in continuous discrete mixed; do
  optiforge develop configs/develop_$c.yaml > results/develop_$c.log 2>&1
done
echo done > results/ALL_DONE

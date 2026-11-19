# Optimizer comparison (surrogate, equal budget)

Per case/family/optimizer, over all theta_s: mean best robust cost (capped at 1000), success-class rate, mean iterations to first feasible (cost < 100), mean evals per seed.

| case | family | optimizer | mean best cost | success rate | iters to feasible | evals/seed |
|---|---|---|---|---|---|---|
| rc01 | brake | cem | 14.4 | 86% | 1.43 | 384 |
| rc01 | brake | icem | 14.4 | 86% | 2.00 | 225 |
| rc01 | brake | mppi | 14.4 | 86% | 1.29 | 384 |
| rc01 | surge_brake | cem | 150.8 | 86% | 2.08 | 384 |
| rc01 | surge_brake | icem | 152.2 | 86% | 1.55 | 225 |
| rc01 | surge_brake | mppi | 149.9 | 86% | 1.42 | 384 |
| rc01 | two_stage | cem | 14.4 | 86% | 1.08 | 384 |
| rc01 | two_stage | icem | 14.4 | 86% | 1.08 | 225 |
| rc01 | two_stage | mppi | 14.4 | 86% | 1.14 | 384 |
| rc03 | brake | icem | 131.0 | 88% | 1.00 | 225 |
| rc03 | keep | icem | 875.0 | 12% | 1.00 | 225 |
| rc03 | lane_change | icem | 753.8 | 12% | 1.00 | 225 |
| rc05 | abort | cem | 1.8 | 100% | 1.00 | 384 |
| rc05 | abort | icem | 1.8 | 100% | 1.00 | 225 |
| rc05 | abort | mppi | 1.8 | 100% | 1.00 | 384 |
| rc05 | abort_surge | cem | 1.8 | 100% | 1.00 | 384 |
| rc05 | abort_surge | icem | 1.8 | 100% | 1.00 | 225 |
| rc05 | abort_surge | mppi | 1.8 | 100% | 1.00 | 384 |
| rc05 | commit_surge | cem | 36.6 | 0% | 1.00 | 384 |
| rc05 | commit_surge | icem | 36.6 | 0% | 1.00 | 225 |
| rc05 | commit_surge | mppi | 36.6 | 0% | 1.00 | 384 |
| rc09 | brake | cem | 352.8 | 67% | 1.88 | 384 |
| rc09 | brake | icem | 352.9 | 67% | 2.25 | 225 |
| rc09 | brake | mppi | 352.9 | 67% | 1.62 | 384 |
| rc09 | surge_brake | cem | 352.8 | 67% | 1.38 | 384 |
| rc09 | surge_brake | icem | 352.8 | 67% | 1.50 | 225 |
| rc09 | surge_brake | mppi | 353.0 | 67% | 1.50 | 384 |
| rc09 | two_stage | cem | 352.8 | 67% | 1.38 | 384 |
| rc09 | two_stage | icem | 353.0 | 67% | 1.50 | 225 |
| rc09 | two_stage | mppi | 353.0 | 67% | 1.38 | 384 |
| rc13 | pre_slow | icem | 59.6 | 0% | 1.00 | 225 |
| rc13 | reachable_set | icem | 174.3 | 25% | nan | 225 |
| rc13 | react | icem | 60.4 | 25% | 1.25 | 225 |

#!/usr/bin/env bash
# Submit CARLA verification of a results_free candidate file (local helper, uses the private Slurm scripts in slurm/). Run from robotproject/.
# usage: bash results_free/submit.sh <batch dir under results_free/rc04/search> <nshards> [amd]      e.g.  bash results_free/submit.sh free_pilot 6
# Any GPU type is accepted (--gres=gpu:1): the queue for a single GPU type was days long when this was written.
set -euo pipefail
B=${1:?}; N=${2:-1}; MODE=${3:-gpu}
mkdir -p slurm_logs
for i in $(seq 0 $((N-1))); do
  if [ "$MODE" = amd ]; then
    CASE=rc04 SHARD=$i NSHARDS=$N CANDS=results_free/rc04/search/$B/candidates.json OUT=results_free/rc04/carla/${B}_amd sbatch --export=ALL --job-name=f_${B}_amd_$i slurm/verify_amd.sbatch
  else
    CASE=rc04 SHARD=$i NSHARDS=$N CANDS=results_free/rc04/search/$B/candidates.json OUT=results_free/rc04/carla/$B sbatch --export=ALL --gres=${GRES:-gpu:1} ${DEP:+--dependency=afterok:$DEP} --job-name=f_${B}_$i slurm/verify.sbatch
  fi
done

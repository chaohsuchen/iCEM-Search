#!/bin/bash
# shard_loop.sh CMD RC BATCH K PORT [EXTRA...]: run one carla_verify shard on a fresh server until nothing is left.
# A server crash, an OOM kill or carla_verify's own exit at high server RSS aborts the shard; the next round resumes.
cmd=$1; rc=$2; b=$3; k=$4; port=$5; shift 5
cd ~/rc
for i in $(seq 1 40); do
  bash ~/rc/restart_srv.sh $port
  ~/rcenv/bin/python carla_rc/teachers/carla_verify.py $cmd --rc $rc --batch $b --shard $k --of 4 --port $port \
      --out ~/rc/results_search "$@" > /tmp/sl_$port.log 2>&1 < /dev/null
  cat /tmp/sl_$port.log
  grep -q " 0 to run" /tmp/sl_$port.log && break
done

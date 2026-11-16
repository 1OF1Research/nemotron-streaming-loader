#!/bin/bash
# Card 13 stage 2 — bounded retention proof at the PROBE shape (BUDGET=0,
# 4 threads): retention cap 9 GiB in the cell (retention law: decode-step
# sets pinned, prefill touches not retained), daemon core 0.5 GB.
set -u
cd /home/agent/hf-weekend/pool-h/card13
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/home/agent/hf-weekend/3a-rig/venv/bin/python
CKPT=/home/agent/hf-weekend/pool-h/nemotron-3-nano-30b-a3b-bf16
ARM=/home/agent/hf-weekend/pool-h/h1/arm-patched
R=runs
L=$R/stage2d.log
echo "STAGE2D START $(date -Is) hostname=$(hostname)" > $L
free -g >> $L
pkill -f expert-loader; sleep 1
rm -f loader.sock
./loader-rs/target/release/expert-loader "$CKPT" /home/agent/hf-weekend/pool-h/card13/loader.sock 500000000 > $R/daemon-stage2d.log 2>&1 &
sleep 3
head -2 $R/daemon-stage2d.log >> $L
echo "ARM4 loader probe-shape retainlaw cap9 max_new=6 $(date -Is)" >> $L
$PY loadercell4.py --arm-dir $ARM --out $R/load-probe-retain9.json --max-new 6 --retain --retain-gib 9.0 >> $R/load-probe-retain9.log 2>&1
echo "retain exit=$? $(date -Is)" >> $L
pkill -f expert-loader
echo "STAGE2D ALL DONE $(date -Is)" >> $L

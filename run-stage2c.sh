#!/bin/bash
# Card 13 stage 2, third attempt: pair arm with retention capped at 2.0 GiB.
set -u
cd /home/agent/hf-weekend/pool-h/card13
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/home/agent/hf-weekend/3a-rig/venv/bin/python
CKPT=/home/agent/hf-weekend/pool-h/nemotron-3-nano-30b-a3b-bf16
ARM=/home/agent/hf-weekend/pool-h/h1/arm-patched
R=runs
L=$R/stage2c.log
echo "STAGE2C START $(date -Is) hostname=$(hostname)" > $L
free -g >> $L
pkill -f expert-loader; sleep 1
rm -f loader.sock
./loader-rs/target/release/expert-loader "$CKPT" /home/agent/hf-weekend/pool-h/card13/loader.sock 2000000000 > $R/daemon-stage2c.log 2>&1 &
sleep 3
head -2 $R/daemon-stage2c.log >> $L
echo "ARM2 loader H1-shape retain capped 2.0GiB max_new=6 $(date -Is)" >> $L
$PY loadercell3.py --arm-dir $ARM --out $R/load-h1shape-6.json --max-new 6 --threads 8 --budget-gib 11 --retain --retain-gib 2.0 >> $R/load-h1shape-6.log 2>&1
echo "load exit=$? $(date -Is)" >> $L
pkill -f expert-loader
echo "STAGE2C ALL DONE $(date -Is)" >> $L

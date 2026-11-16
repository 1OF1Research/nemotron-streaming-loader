#!/bin/bash
# Card 13 stage 2, second attempt: loader arms with daemon budget sized to
# the H1 shape (2 GB daemon core; the cell carries the retention).
set -u
cd /home/agent/hf-weekend/pool-h/card13
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/home/agent/hf-weekend/3a-rig/venv/bin/python
CKPT=/home/agent/hf-weekend/pool-h/nemotron-3-nano-30b-a3b-bf16
ARM=/home/agent/hf-weekend/pool-h/h1/arm-patched
R=runs
L=$R/stage2b.log
echo "STAGE2B START $(date -Is) hostname=$(hostname)" > $L
free -g >> $L
pkill -f expert-loader; sleep 1
rm -f loader.sock
./loader-rs/target/release/expert-loader "$CKPT" /home/agent/hf-weekend/pool-h/card13/loader.sock 2000000000 > $R/daemon-stage2b.log 2>&1 &
sleep 3
head -2 $R/daemon-stage2b.log >> $L

echo "ARM2 loader H1-shape retain unbounded max_new=6 $(date -Is)" >> $L
$PY loadercell3.py --arm-dir $ARM --out $R/load-h1shape-6.json --max-new 6 --threads 8 --budget-gib 11 --retain >> $R/load-h1shape-6.log 2>&1
echo "load exit=$? $(date -Is)" >> $L
free -g >> $L

# free the pair daemon's core before the identity run starts its own
pkill -f expert-loader; sleep 2
rm -f loader.sock
./loader-rs/target/release/expert-loader "$CKPT" /home/agent/hf-weekend/pool-h/card13/loader.sock 2000000000 > $R/daemon-stage2b-ident.log 2>&1 &
sleep 3

echo "ARM3 loader H1-shape retain capped 1.5GiB max_new=32 identity $(date -Is)" >> $L
$PY loadercell3.py --arm-dir $ARM --out $R/load-h1shape-32.json --max-new 32 --threads 8 --budget-gib 11 --retain --retain-gib 1.5 >> $R/load-h1shape-32.log 2>&1
echo "ident exit=$? $(date -Is)" >> $L

pkill -f expert-loader
echo "STAGE2B ALL DONE $(date -Is)" >> $L

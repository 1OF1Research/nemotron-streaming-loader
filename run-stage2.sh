#!/bin/bash
# Card 13 stage 2 — H1-shape measured pair + identity run. Serial, never concurrent.
set -u
cd /home/agent/hf-weekend/pool-h/card13
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/home/agent/hf-weekend/3a-rig/venv/bin/python
CKPT=/home/agent/hf-weekend/pool-h/nemotron-3-nano-30b-a3b-bf16
ARM=/home/agent/hf-weekend/pool-h/h1/arm-patched
R=runs
L=$R/stage2.log
echo "STAGE2 START $(date -Is) hostname=$(hostname)" > $L
uptime >> $L; free -g >> $L; df -h / | tail -1 >> $L
ps aux | grep -E "python|expert-loader" | grep -v grep >> $L || true

# daemon fresh, 9 GB budget (same as stage 1)
pkill -f expert-loader; sleep 1
rm -f loader.sock
./loader-rs/target/release/expert-loader "$CKPT" /home/agent/hf-weekend/pool-h/card13/loader.sock 9000000000 > $R/daemon-stage2.log 2>&1 &
sleep 3
head -2 $R/daemon-stage2.log >> $L

echo "ARM1 shipped control H1-shape max_new=6 $(date -Is)" >> $L
$PY streamprobe2.py --arm-dir $ARM --out $R/ship-h1shape-6.json --max-new 6 --threads 8 --budget-gib 11 >> $R/ship-h1shape-6.log 2>&1
echo "ship exit=$? $(date -Is)" >> $L

echo "ARM2 loader H1-shape retain max_new=6 $(date -Is)" >> $L
$PY loadercell2.py --arm-dir $ARM --out $R/load-h1shape-6.json --max-new 6 --threads 8 --budget-gib 11 --retain >> $R/load-h1shape-6.log 2>&1
echo "load exit=$? $(date -Is)" >> $L

echo "ARM3 loader H1-shape retain max_new=32 identity $(date -Is)" >> $L
$PY loadercell2.py --arm-dir $ARM --out $R/load-h1shape-32.json --max-new 32 --threads 8 --budget-gib 11 --retain >> $R/load-h1shape-32.log 2>&1
echo "ident exit=$? $(date -Is)" >> $L

pkill -f expert-loader
echo "STAGE2 ALL DONE $(date -Is)" >> $L

#!/bin/bash
# Standalone eval watcher — decoupled from training.
# Watches the training log; every N epochs runs the FULL evaluate.py
# (P1 MPJPE + P2 P-MPJPE + flip-TTA + per-action) on the current best ckpt.
# Runs the heavy occlusion study once at the end (it is ~40 full-test passes;
# set OCCL_EVERY>0 to also run it periodically, but it roughly doubles wall-clock).
#
# Usage: bash scripts/auto_eval.sh [EVERY=10] [TAG=kinfk_cpn_sota] [OCCL_EVERY=0]
set -u
cd /opt/dlami/nvme/timymamba
export PYTHONPATH=/opt/dlami/nvme/timymamba CUDA_VISIBLE_DEVICES=0
PY=/opt/dlami/nvme/envs/bsmamba/bin/python
EVERY=${1:-10}
TAG=${2:-kinfk_cpn_sota}
OCCL_EVERY=${3:-0}
CFG=configs/cpn_tiny_sota.yaml
CKPT=checkpoints/best_${TAG}.pth
TLOG=logs/${TAG}.log
OUT=logs/auto_eval_${TAG}.log

echo "=== auto_eval watching $TLOG | evaluate every $EVERY ep | occl_every=$OCCL_EVERY | $(date +%F_%H:%M:%S) ===" | tee -a "$OUT"

run_eval () {  # $1 = epoch label
    echo "" | tee -a "$OUT"
    echo "########## EVAL @ epoch $1  ($(date +%H:%M:%S)) ##########" | tee -a "$OUT"
    $PY evaluate.py --config "$CFG" --checkpoint "$CKPT" 2>/dev/null \
        | grep -aE "Protocol|Ours|Test frames|^    [A-Z]" | tee -a "$OUT"
}
run_occl () {  # $1 = epoch label
    echo "----- OCCLUSION @ epoch $1 ($(date +%H:%M:%S)) -----" | tee -a "$OUT"
    $PY scripts/occlusion_eval.py --config "$CFG" --checkpoint "$CKPT" 2>/dev/null \
        | grep -aE "sigma|conf-aware|joint|Occlusion" | tee -a "$OUT"
}

last=0
while true; do
    ep=$(grep -aoE "Epoch +[0-9]+/" "$TLOG" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
    ep=${ep:-0}
    milestone=$(( (ep / EVERY) * EVERY ))
    if [ "$milestone" -ge "$EVERY" ] && [ "$milestone" -gt "$last" ] && [ -f "$CKPT" ]; then
        run_eval "$milestone"
        if [ "$OCCL_EVERY" -gt 0 ] && [ $(( milestone % OCCL_EVERY )) -eq 0 ]; then
            run_occl "$milestone"
        fi
        last=$milestone
    fi
    # training finished?  -> final full eval + occlusion, then exit
    if ! pgrep -f "bin/python train.py --config ${CFG}" >/dev/null 2>&1; then
        sleep 10
        if ! pgrep -f "bin/python train.py --config ${CFG}" >/dev/null 2>&1; then
            echo "" | tee -a "$OUT"
            echo "########## TRAINING ENDED — FINAL eval + occlusion ($(date +%H:%M:%S)) ##########" | tee -a "$OUT"
            run_eval "FINAL-best"
            run_occl "FINAL-best"
            break
        fi
    fi
    sleep 60
done
echo "=== auto_eval finished $(date +%H:%M:%S) ===" | tee -a "$OUT"

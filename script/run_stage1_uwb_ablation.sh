#!/usr/bin/env bash
#
# Stage-1 UWB ablation: 3 encoders x 4 seq_len x 4 bce_weight = 48 experiments
#
# Phase 1: train all 48 configs
# Phase 2: run analysis once
#
set -euo pipefail

SAVE_DIR="${1:-output}"
START_IDX="${2:-0}"  # resume from Nth config (0-based)
PHASE="${3:-all}"    # train / analysis / all

export PYTHONIOENCODING=utf-8

CONFIG_DIR="experiments/ugtrack"

# Generate all 48 config names (encoder, seq_len, weight_suffix)
configs=()
for enc in mlp gru tcn; do
    for sl in 1 3 5 10; do
        for ws in 01 025 05 10; do
            configs+=("${enc}:${sl}:${ws}")
        done
    done
done

total="${#configs[@]}"

# ============================================
# Phase 1: Train
# ============================================
if [[ "$PHASE" == "all" || "$PHASE" == "train" ]]; then
    for i in $(seq $START_IDX $((total - 1))); do
        IFS=':' read -r enc sl ws <<< "${configs[$i]}"
        name="s1_${enc}_t${sl}_bce${ws}"
        config_path="${CONFIG_DIR}/${name}.yaml"

        echo
        echo "=============================================================="
        echo "[$((i+1))/${total}] Training: ${name} (encoder=${enc}, T=${sl}, W=0.${ws})"
        echo "=============================================================="

        python tracking/train_uwb.py \
            --config "$config_path" \
            --save_dir "$SAVE_DIR"
    done
fi

# ============================================
# Phase 2: Analyze
# ============================================
if [[ "$PHASE" == "all" || "$PHASE" == "analysis" ]]; then
    echo
    echo "=============================================================="
    echo "All training done. Running analysis..."
    echo "=============================================================="

    python tracking/analysis_uwb_results.py
fi

echo
echo "Done. Phase=${PHASE}, configs processed: ${total}"

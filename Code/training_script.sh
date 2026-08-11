#!/bin/bash
# Run all 5 stress-aware AgroNT training configurations sequentially.
# Stops on first failure to avoid wasting GPU time on dependent runs.

set -e  # Exit on first error

SCRIPT="stress_aware_agront.py"
GPUS=4

echo "============================================================"
echo "Starting all training runs at $(date)"
echo "============================================================"

# 1. Prompt tuning only
echo ""
echo "[1/5] Prompt tuning only"
echo "------------------------------------------------------------"
torchrun --nproc_per_node=$GPUS $SCRIPT \
    --version "prompt_tuning_v1"

# 2. Prompt tuning + LoRA
echo ""
echo "[2/5] Prompt tuning + LoRA"
echo "------------------------------------------------------------"
torchrun --nproc_per_node=$GPUS $SCRIPT \
    --use_lora \
    --version "prompt_tuning_lora_v1"

# 3. Prompt tuning + DoRA
echo ""
echo "[3/5] Prompt tuning + DoRA"
echo "------------------------------------------------------------"
torchrun --nproc_per_node=$GPUS $SCRIPT \
    --use_lora --use_dora \
    --version "prompt_tuning_dora_v1"

# 4. LoRA only
echo ""
echo "[4/5] LoRA only"
echo "------------------------------------------------------------"
torchrun --nproc_per_node=$GPUS $SCRIPT \
    --use_lora --no_prompt_tuning \
    --version "lora_only_v1"

# 5. Prompt tuning + rsLoRA
echo ""
echo "[5/5] Prompt tuning + rsLoRA"
echo "------------------------------------------------------------"
torchrun --nproc_per_node=$GPUS $SCRIPT \
    --use_lora --use_rslora \
    --version "prompt_tuning_rslora_v1"

echo ""
echo "============================================================"
echo "All runs completed at $(date)"
echo "============================================================"
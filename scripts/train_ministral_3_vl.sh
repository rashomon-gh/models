#!/bin/bash

# Script to train Ministral-3-vl model with custom configurations
# This script uses the current Python virtual environment (activated with 'uv' or manually)
#
# Usage examples:
#   ./scripts/train_ministral_3_vl.sh
#   ./scripts/train_ministral_3_vl.sh --load-in-4bit --max-steps 100
#   ./scripts/train_ministral_3_vl.sh --push-to-hub --hub-model-id username/model

set -e  # Exit on error
set -u  # Exit on undefined variable

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Create logs directory if it doesn't exist
mkdir -p logs

# Default values for training
MODEL_NAME="unsloth/Ministral-3-3B-Instruct-2512"
LOAD_IN_4BIT="false"
USE_GRADIENT_CHECKPOINTING="unsloth"
R=32
LORA_ALPHA=32
LORA_DROPOUT=0

DATASET_NAME="unsloth/LaTeX_OCR"
DATASET_SPLIT="train"
INSTRUCTION="Write the LaTeX representation for this image."

PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2
WARMUP_STEPS=5
MAX_STEPS=30
NUM_TRAIN_EPOCHS=""
LEARNING_RATE="2e-4"
LOGGING_STEPS=1
OPTIM="adamw_8bit"
WEIGHT_DECAY="0.001"
LR_SCHEDULER_TYPE="linear"
SEED=3407
OUTPUT_DIR="outputs"
REPORT_TO="tensorboard"
MAX_LENGTH=2048

SAVE_LOCAL_16BIT="false"
SAVE_LOCAL_PATH="unsloth_finetune"
PUSH_TO_HUB="true"
HUB_MODEL_ID="ministral-3-vl"
PUSH_GGUF="true"
GGUF_MODEL_ID="ministral-3-vl-gguf"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --load-in-4bit)
            LOAD_IN_4BIT="true"
            shift
            ;;
        --use-gradient-checkpointing)
            USE_GRADIENT_CHECKPOINTING="$2"
            shift 2
            ;;
        --r)
            R="$2"
            shift 2
            ;;
        --lora-alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        --lora-dropout)
            LORA_DROPOUT="$2"
            shift 2
            ;;
        --dataset-name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --dataset-split)
            DATASET_SPLIT="$2"
            shift 2
            ;;
        --instruction)
            INSTRUCTION="$2"
            shift 2
            ;;
        --per-device-train-batch-size)
            PER_DEVICE_TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --gradient-accumulation-steps)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --warmup-steps)
            WARMUP_STEPS="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --num-train-epochs)
            NUM_TRAIN_EPOCHS="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --logging-steps)
            LOGGING_STEPS="$2"
            shift 2
            ;;
        --optim)
            OPTIM="$2"
            shift 2
            ;;
        --weight-decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --lr-scheduler-type)
            LR_SCHEDULER_TYPE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --report-to)
            REPORT_TO="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --save-local-16bit)
            SAVE_LOCAL_16BIT="true"
            shift
            ;;
        --no-save-local)
            SAVE_LOCAL_16BIT="false"
            shift
            ;;
        --save-local-path)
            SAVE_LOCAL_PATH="$2"
            shift 2
            ;;
        --push-to-hub)
            PUSH_TO_HUB="true"
            shift
            ;;
        --hub-model-id)
            HUB_MODEL_ID="$2"
            shift 2
            ;;
        --push-gguf)
            PUSH_GGUF="true"
            shift
            ;;
        --gguf-model-id)
            GGUF_MODEL_ID="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Script to train Ministral-3-vl model with custom configurations."
            echo ""
            echo "Options:"
            echo "  --model-name NAME                  Model name/path (default: unsloth/Ministral-3-3B-Instruct-2512)"
            echo "  --load-in-4bit                    Load in 4-bit quantization"
            echo "  --use-gradient-checkpointing STRATEGY  Gradient checkpointing (default: unsloth)"
            echo "  --r RANK                           LoRA rank (default: 32)"
            echo "  --lora-alpha ALPHA                  LoRA alpha (default: 32)"
            echo "  --lora-dropout DROPOUT              LoRA dropout (default: 0)"
            echo "  --dataset-name NAME                 Dataset name (default: unsloth/LaTeX_OCR)"
            echo "  --dataset-split SPLIT               Dataset split (default: train)"
            echo "  --instruction INSTRUCTION             Instruction for model"
            echo "  --per-device-train-batch-size SIZE   Batch size (default: 4)"
            echo "  --gradient-accumulation-steps STEPS  Gradient accumulation (default: 2)"
            echo "  --warmup-steps STEPS               Warmup steps (default: 5)"
            echo "  --max-steps STEPS                  Max training steps (default: 30)"
            echo "  --num-train-epochs EPOCHS           Number of epochs"
            echo "  --learning-rate RATE                Learning rate (default: 2e-4)"
            echo "  --logging-steps STEPS               Logging frequency (default: 1)"
            echo "  --optim OPTIMIZER                  Optimizer (default: adamw_8bit)"
            echo "  --weight-decay DECAY               Weight decay (default: 0.001)"
            echo "  --lr-scheduler-type TYPE           LR scheduler (default: linear)"
            echo "  --seed SEED                       Random seed (default: 3407)"
            echo "  --output-dir DIR                   Output directory (default: outputs)"
            echo "  --report-to TARGET                 Metrics reporting (default: tensorboard)"
            echo "  --max-length LENGTH                Max sequence length (default: 2048)"
            echo "  --save-local-16bit                 Save model locally (default: true)"
            echo "  --no-save-local                   Don't save model locally"
            echo "  --save-local-path PATH             Local save path (default: unsloth_finetune)"
            echo "  --push-to-hub                     Push to Hugging Face Hub"
            echo "  --hub-model-id ID                  Hub model ID"
            echo "  --push-gguf                        Push GGUF to Hub"
            echo "  --gguf-model-id ID                GGUF model ID"
            echo "  --help                            Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --load-in-4bit --max-steps 100"
            echo "  $0 --push-to-hub --hub-model-id username/model"
            echo "  $0 --num-train-epochs 3 --learning-rate 1e-4"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build Python arguments
PYTHON_ARGS="--model-name '$MODEL_NAME'"

if [ "$LOAD_IN_4BIT" = "true" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --load-in-4bit"
fi

PYTHON_ARGS="$PYTHON_ARGS --use-gradient-checkpointing '$USE_GRADIENT_CHECKPOINTING'"
PYTHON_ARGS="$PYTHON_ARGS --r $R"
PYTHON_ARGS="$PYTHON_ARGS --lora-alpha $LORA_ALPHA"
PYTHON_ARGS="$PYTHON_ARGS --lora-dropout $LORA_DROPOUT"
PYTHON_ARGS="$PYTHON_ARGS --dataset-name '$DATASET_NAME'"
PYTHON_ARGS="$PYTHON_ARGS --dataset-split '$DATASET_SPLIT'"
PYTHON_ARGS="$PYTHON_ARGS --instruction '$INSTRUCTION'"
PYTHON_ARGS="$PYTHON_ARGS --per-device-train-batch-size $PER_DEVICE_TRAIN_BATCH_SIZE"
PYTHON_ARGS="$PYTHON_ARGS --gradient-accumulation-steps $GRADIENT_ACCUMULATION_STEPS"
PYTHON_ARGS="$PYTHON_ARGS --warmup-steps $WARMUP_STEPS"
PYTHON_ARGS="$PYTHON_ARGS --max-steps $MAX_STEPS"

if [ -n "$NUM_TRAIN_EPOCHS" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --num-train-epochs $NUM_TRAIN_EPOCHS"
fi

PYTHON_ARGS="$PYTHON_ARGS --learning-rate $LEARNING_RATE"
PYTHON_ARGS="$PYTHON_ARGS --logging-steps $LOGGING_STEPS"
PYTHON_ARGS="$PYTHON_ARGS --optim '$OPTIM'"
PYTHON_ARGS="$PYTHON_ARGS --weight-decay $WEIGHT_DECAY"
PYTHON_ARGS="$PYTHON_ARGS --lr-scheduler-type '$LR_SCHEDULER_TYPE'"
PYTHON_ARGS="$PYTHON_ARGS --seed $SEED"
PYTHON_ARGS="$PYTHON_ARGS --output-dir '$OUTPUT_DIR'"
PYTHON_ARGS="$PYTHON_ARGS --report-to '$REPORT_TO'"
PYTHON_ARGS="$PYTHON_ARGS --max-length $MAX_LENGTH"

if [ "$SAVE_LOCAL_16BIT" = "true" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --save-local-16bit"
fi

PYTHON_ARGS="$PYTHON_ARGS --save-local-path '$SAVE_LOCAL_PATH'"

if [ "$PUSH_TO_HUB" = "true" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --push-to-hub"
fi

PYTHON_ARGS="$PYTHON_ARGS --hub-model-id '$HUB_MODEL_ID'"

if [ "$PUSH_GGUF" = "true" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --push-gguf"
fi

PYTHON_ARGS="$PYTHON_ARGS --gguf-model-id '$GGUF_MODEL_ID'"

# Display configuration
echo "================================"
echo "Training Configuration"
echo "================================"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Max Steps: $MAX_STEPS"
echo "Batch Size: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Output Dir: $OUTPUT_DIR"
echo "================================"
echo ""

# Run training
echo "Starting training..."
echo ""

python -m models.ministral_3_vl $PYTHON_ARGS

echo ""
echo "Training completed!"

# Fine-tuning

A collection of fine-tuning scripts using LoRA/PEFT methods.

## Installation

This project uses `uv` for dependency management.

Create and activate the virtual environment:

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Configuration

Create a `.env` file in project root based on `.env.example`:

```bash
cp .env.example .env
```

Update the `.env` file with your Hugging Face token and user name if you plan to push models to the hub:

```
HUGGINGFACE_TOKEN=<your-huggingface-token>
HUGGINGFACE_USERNAME=<your-huggingface-username>
```

## Example Usage

For each model script in `/models`, there's a bash script in `/scripts` which can be run to train and push a model to the huggingface hub. Below is an example for `ministral-3-vl`.


```bash
# Make script executable (first time only)
chmod +x scripts/train_ministral_3_vl.sh

# Run with default settings
./scripts/train_ministral_3_vl.sh
```


```bash
# Training with 4-bit quantization (reduced memory)
./scripts/train_ministral_3_vl.sh --load-in-4bit

# Training for more steps
./scripts/train_ministral_3_vl.sh --max-steps 100

# Training for multiple epochs
./scripts/train_ministral_3_vl.sh --num-train-epochs 3

# Custom batch size and learning rate
./scripts/train_ministral_3_vl.sh \
    --per-device-train-batch-size 2 \
    --learning-rate 1e-4
```

```bash
# Save locally only
./scripts/train_ministral_3_vl.sh \
    --save-local-16bit \
    --save-local-path my_model

# Push to Hugging Face Hub
./scripts/train_ministral_3_vl.sh \
    --push-to-hub \
    --hub-model-id your-model-name

# Push everything (local + hub + GGUF)
./scripts/train_ministral_3_vl.sh \
    --save-local-16bit \
    --save-local-path my_model \
    --push-to-hub \
    --hub-model-id your-model-name \
    --push-gguf \
    --gguf-model-id your-model-gguf
```

For more config options, check ``

## Available Scripts

- `scripts/train_ministral_3_vl.sh` - Training script for Ministral-3-vl model

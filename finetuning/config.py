from pydantic import BaseModel, SecretStr
from typing import List, Optional
from finetuning.settings import settings


class ModelConfig(BaseModel):
    """Configuration for model loading and LoRA setup."""

    model_name: str = "unsloth/Ministral-3-3B-Instruct-2512"
    load_in_4bit: bool = False
    use_gradient_checkpointing: str = "unsloth"

    # PEFT/LoRA parameters
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    r: int = 32
    lora_alpha: int = 32
    lora_dropout: int = 0
    bias: str = "none"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Optional[dict] = None


class DatasetConfig(BaseModel):
    """Configuration for dataset loading and processing."""

    dataset_name: str = "unsloth/LaTeX_OCR"
    dataset_split: str = "train"
    instruction: str = "Write the LaTeX representation for this image."


class TrainingConfig(BaseModel):
    """Configuration for training parameters."""

    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 5
    max_steps: int = 30
    num_train_epochs: Optional[int] = None
    learning_rate: float = 2e-4
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.001
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    output_dir: str = "outputs"
    report_to: str = "tensorboard"
    max_length: int = 2048


class ExportConfig(BaseModel):
    """Configuration for model export options."""

    save_local_16bit: bool = False
    save_local_path: str = "unsloth_finetune"

    push_to_hub: bool = False
    hub_model_id: str = "YOUR_USERNAME/unsloth_finetune"
    hub_token: Optional[SecretStr] = None

    push_gguf: bool = False
    gguf_model_id: str = "hf/unsloth_finetune"
    gguf_quantization_methods: List[str] = ["q4_k_m", "q8_0", "q5_k_m"]

    def __init__(self, **data):
        # Use settings token if not provided
        if "hub_token" not in data or data["hub_token"] is None:
            data["hub_token"] = settings.huggingface_token
        super().__init__(**data)

"""
Ministral-3-vl model training module.

This module provides a self-contained implementation for fine-tuning
Ministral-3-vl vision-language model using Unsloth.
"""

import sys
import argparse
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from unsloth import is_bf16_supported
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset as HFDataset
from typing import Dict, List, Any
from loguru import logger

from finetuning import ModelConfig, DatasetConfig, TrainingConfig, ExportConfig
from finetuning.settings import settings


class Ministral3VLTrainer:
    """Trainer for Ministral-3-vl vision-language model."""

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
        export_config: ExportConfig,
    ):
        """
        Initialize the Ministral-3-vl trainer.

        Args:
            model_config: Configuration for model loading and LoRA setup
            dataset_config: Configuration for dataset loading and processing
            training_config: Configuration for training parameters
            export_config: Configuration for model export
        """
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.training_config = training_config
        self.export_config = export_config

        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset = None

    def load_model(self):
        """Load the Ministral-3-vl model and tokenizer."""
        logger.info(f"Loading model: {self.model_config.model_name}")
        logger.debug(f"Load in 4-bit: {self.model_config.load_in_4bit}")
        logger.debug(
            f"Use gradient checkpointing: {self.model_config.use_gradient_checkpointing}"
        )

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            self.model_config.model_name,
            load_in_4bit=self.model_config.load_in_4bit,
            use_gradient_checkpointing=self.model_config.use_gradient_checkpointing,
        )

        logger.success("Model and tokenizer loaded successfully")

    def apply_lora(self):
        """Apply LoRA/PEFT to the Ministral-3-vl model."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model or tokenizer not loaded. Call load_model() first."
            )

        logger.info("Applying LoRA/PEFT to model")
        logger.debug(f"LoRA rank (r): {self.model_config.r}")
        logger.debug(f"LoRA alpha: {self.model_config.lora_alpha}")
        logger.debug(
            f"Finetune vision layers: {self.model_config.finetune_vision_layers}"
        )
        logger.debug(
            f"Finetune language layers: {self.model_config.finetune_language_layers}"
        )

        self.model = FastVisionModel.get_peft_model(
            self.model,
            finetune_vision_layers=self.model_config.finetune_vision_layers,
            finetune_language_layers=self.model_config.finetune_language_layers,
            finetune_attention_modules=self.model_config.finetune_attention_modules,
            finetune_mlp_modules=self.model_config.finetune_mlp_modules,
            r=self.model_config.r,
            lora_alpha=self.model_config.lora_alpha,
            lora_dropout=self.model_config.lora_dropout,
            bias=self.model_config.bias,
            random_state=self.model_config.random_state,
            use_rslora=self.model_config.use_rslora,
            loftq_config=self.model_config.loftq_config,
        )

        logger.success("LoRA/PEFT applied successfully")

    def load_dataset(self):
        """Load and prepare the LaTeX OCR dataset."""
        logger.info(f"Loading dataset: {self.dataset_config.dataset_name}")
        logger.debug(f"Dataset split: {self.dataset_config.dataset_split}")

        self.dataset = load_dataset(
            self.dataset_config.dataset_name, split=self.dataset_config.dataset_split
        )

        # Log dataset size (only if it has __len__)
        try:
            dataset_size = len(self.dataset)  # type: ignore[arg-type]
            logger.info(f"Dataset loaded with {dataset_size} samples")
        except TypeError:
            logger.info("Dataset loaded successfully")
            logger.debug("Dataset is iterable, size unknown")

        logger.debug("Converting to conversation format...")

        # Convert to conversation format
        converted_dataset = [
            self._convert_to_conversation(sample) for sample in self.dataset
        ]

        # Convert list to Dataset
        self.train_dataset = HFDataset.from_list(converted_dataset)

        logger.success("Dataset prepared successfully")

    def _convert_to_conversation(self, sample: Any) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convert a dataset sample to conversation format.

        Args:
            sample: A sample from the dataset containing image and text

        Returns:
            Dictionary with messages in conversation format
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.dataset_config.instruction},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["text"]}],
            },
        ]
        return {"messages": conversation}

    def prepare_trainer(self):
        """Prepare the SFT trainer for training."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model or tokenizer not loaded. Call load_model() first."
            )

        if not hasattr(self, "train_dataset"):
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")

        # Determine training mode
        if self.training_config.num_train_epochs is not None:
            # Use epochs instead of max_steps
            max_steps_value = -1
            num_train_epochs_value = self.training_config.num_train_epochs
            logger.info(f"Training for {num_train_epochs_value} epochs")
        else:
            # Use max_steps
            max_steps_value = self.training_config.max_steps
            num_train_epochs_value = 1
            logger.info(f"Training for {max_steps_value} steps")

        logger.info("Preparing SFT trainer")
        logger.debug(
            f"Per device batch size: {self.training_config.per_device_train_batch_size}"
        )
        logger.debug(
            f"Gradient accumulation steps: {self.training_config.gradient_accumulation_steps}"
        )
        logger.debug(f"Learning rate: {self.training_config.learning_rate}")

        self.trainer = SFTTrainer(
            model=self.model,
            data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
            train_dataset=self.train_dataset,
            args=SFTConfig(
                per_device_train_batch_size=self.training_config.per_device_train_batch_size,
                gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
                warmup_steps=self.training_config.warmup_steps,
                max_steps=max_steps_value,
                num_train_epochs=num_train_epochs_value,
                learning_rate=self.training_config.learning_rate,
                logging_steps=self.training_config.logging_steps,
                optim=self.training_config.optim,
                fp16=not is_bf16_supported(),
                bf16=is_bf16_supported(),
                weight_decay=self.training_config.weight_decay,
                lr_scheduler_type=self.training_config.lr_scheduler_type,
                seed=self.training_config.seed,
                output_dir=self.training_config.output_dir,
                report_to=self.training_config.report_to,
                # Vision finetuning specific settings
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_length=self.training_config.max_length,
            ),
        )

        logger.success("Trainer prepared successfully")

    def train(self) -> Dict[str, float]:
        """
        Train the model.

        Returns:
            Training statistics
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call prepare_trainer() first.")

        logger.info("Starting training...")
        trainer_stats = self.trainer.train()

        logger.success("Training completed successfully")
        logger.info(f"Training statistics: {trainer_stats}")

        return trainer_stats

    def export_model(self):
        """Export the model according to export configuration."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not initialized.")

        if self.export_config.save_local_16bit:
            logger.info("Saving model locally in 16-bit format")
            self._save_local()
            logger.success(f"Model saved to: {self.export_config.save_local_path}")

        if self.export_config.push_to_hub:
            logger.info("Pushing model to Hugging Face Hub")
            self._push_to_hub()
            logger.success(f"Model pushed to: {self.export_config.hub_model_id}")

        if self.export_config.push_gguf:
            logger.info("Pushing GGUF quantized model to Hub")
            self._push_gguf()
            logger.success(f"GGUF model pushed to: {self.export_config.gguf_model_id}")

    def _save_local(self):
        """Save model locally in 16-bit format."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not initialized.")

        self.model.save_pretrained_merged(
            self.export_config.save_local_path,
        )

    def _push_to_hub(self):
        """Push model to Hugging Face Hub."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not initialized.")

        token_value = None
        if self.export_config.hub_token is not None:
            token_value = self.export_config.hub_token.get_secret_value()
        else:
            # Use settings token if not provided
            token_value = settings.huggingface_token.get_secret_value()

        self.model.push_to_hub_merged(
            self.export_config.hub_model_id,
            self.tokenizer,
            token=token_value,
        )

    def _push_gguf(self):
        """Push GGUF quantized model to Hugging Face Hub."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not initialized.")

        token_value = None
        if self.export_config.hub_token is not None:
            token_value = self.export_config.hub_token.get_secret_value()
        else:
            # Use settings token if not provided
            token_value = settings.huggingface_token.get_secret_value()

        self.model.push_to_hub_gguf(
            self.export_config.gguf_model_id,
            self.tokenizer,
            quantization_method=self.export_config.gguf_quantization_methods,
            token=token_value,
        )

    def run_full_pipeline(self) -> Dict[str, float]:
        """
        Run the complete training and export pipeline.

        Returns:
            Training statistics
        """
        logger.info("=" * 50)
        logger.info("Starting Ministral-3-vl training pipeline")
        logger.info("=" * 50)
        logger.info("")

        self.load_model()
        self.apply_lora()
        self.load_dataset()
        self.prepare_trainer()
        stats = self.train()
        self.export_model()

        logger.info("")
        logger.info("=" * 50)
        logger.success("Training pipeline completed successfully!")
        logger.info("=" * 50)

        return stats


if __name__ == "__main__":
    # Configure logger
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        "logs/training.log",
        rotation="500 MB",
        retention="10 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train Ministral-3-vl vision-language model with custom configurations"
    )

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="unsloth/Ministral-3-3B-Instruct-2512",
        help="Name/path of model to load",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization (reduces memory usage)",
    )
    parser.add_argument(
        "--use-gradient-checkpointing",
        type=str,
        default="unsloth",
        help="Gradient checkpointing strategy (default: unsloth)",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=32,
        help="LoRA rank (default: 32)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter (default: 32)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=int,
        default=0,
        help="LoRA dropout (default: 0)",
    )
    parser.add_argument(
        "--finetune-vision-layers",
        action="store_true",
        default=True,
        help="Finetune vision layers (default: True)",
    )
    parser.add_argument(
        "--finetune-language-layers",
        action="store_true",
        default=True,
        help="Finetune language layers (default: True)",
    )
    parser.add_argument(
        "--finetune-attention-modules",
        action="store_true",
        default=True,
        help="Finetune attention modules (default: True)",
    )
    parser.add_argument(
        "--finetune-mlp-modules",
        action="store_true",
        default=True,
        help="Finetune MLP modules (default: True)",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="unsloth/LaTeX_OCR",
        help="Name/path of dataset to load",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Write the LaTeX representation for this image.",
        help="Instruction for the model",
    )

    # Training arguments
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=4,
        help="Per device training batch size (default: 4)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of warmup steps (default: 5)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum number of training steps (default: 30)",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides max-steps)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Logging frequency in steps (default: 1)",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_8bit",
        help="Optimizer to use (default: adamw_8bit)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.001,
        help="Weight decay (default: 0.001)",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="linear",
        help="Learning rate scheduler type (default: linear)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed (default: 3407)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)",
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default="tensorboard",
        help="Where to report metrics (default: tensorboard)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )

    # Export arguments
    parser.add_argument(
        "--save-local-16bit",
        action="store_true",
        help="Save model locally in 16-bit format",
    )
    parser.add_argument(
        "--save-local-path",
        type=str,
        default="unsloth_finetune",
        help="Local path to save model (default: unsloth_finetune)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push model to Hugging Face Hub",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default="YOUR_USERNAME/unsloth_finetune",
        help="Hugging Face Hub model ID",
    )
    parser.add_argument(
        "--push-gguf",
        action="store_true",
        help="Push GGUF quantized model to Hub",
    )
    parser.add_argument(
        "--gguf-model-id",
        type=str,
        default="hf/unsloth_finetune",
        help="GGUF model ID",
    )

    # Parse arguments
    args = parser.parse_args()

    # Configure model
    model_config = ModelConfig(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=args.finetune_language_layers,
        finetune_attention_modules=args.finetune_attention_modules,
        finetune_mlp_modules=args.finetune_mlp_modules,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Configure dataset
    dataset_config = DatasetConfig(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        instruction=args.instruction,
    )

    # Configure training parameters
    training_config = TrainingConfig(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        output_dir=args.output_dir,
        report_to=args.report_to,
        max_length=args.max_length,
    )

    # Configure export options
    export_config = ExportConfig(
        save_local_16bit=args.save_local_16bit,
        save_local_path=args.save_local_path,
        push_to_hub=args.push_to_hub,
        hub_model_id=f"{settings.huggingface_username}/{args.hub_model_id}",
        push_gguf=args.push_gguf,
        gguf_model_id=f"{settings.huggingface_username}/{args.gguf_model_id}",
    )

    # Create trainer instance
    trainer = Ministral3VLTrainer(
        model_config=model_config,
        dataset_config=dataset_config,
        training_config=training_config,
        export_config=export_config,
    )

    # Run the full training pipeline
    trainer_stats = trainer.run_full_pipeline()

"""
Ministral-3-vl model training module.

This module provides a self-contained implementation for fine-tuning the
Ministral-3-vl vision-language model using Unsloth.
"""

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
    # Configure model
    model_config = ModelConfig(
        model_name="unsloth/Ministral-3-3B-Instruct-2512",
        load_in_4bit=False,  # Set to True to reduce memory usage
        use_gradient_checkpointing="unsloth",
        r=32,
        lora_alpha=32,
        lora_dropout=0,
    )

    # Configure the dataset
    dataset_config = DatasetConfig(
        dataset_name="unsloth/LaTeX_OCR",
        dataset_split="train",
        instruction="Write the LaTeX representation for this image.",
    )

    # Configure training parameters
    training_config = TrainingConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        max_steps=30,  # Use num_train_epochs for full training runs
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="tensorboard",
        max_length=2048,
    )

    # Configure export options
    export_config = ExportConfig(
        save_local_16bit=True,
        save_local_path="unsloth_finetune",
        push_to_hub=False,  # Set to True to push to Hugging Face Hub
        hub_model_id="YOUR_USERNAME/unsloth_finetune",
        push_gguf=False,  # Set to True to push GGUF quantized model
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

"""
Example script for training Ministral-3-vl model.

This script demonstrates how to use the refactored Ministral3VLTrainer class.
"""

from finetuning import ModelConfig, DatasetConfig, TrainingConfig, ExportConfig
from models import Ministral3VLTrainer


def main():
    """Main training function."""

    # Configure the model
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
        # hub_model_id="YOUR_USERNAME/unsloth_finetune",
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
    print("Starting training pipeline...")
    print(f"Model: {model_config.model_name}")
    print(f"Dataset: {dataset_config.dataset_name}")
    print(f"Max steps: {training_config.max_steps}")
    print()

    trainer_stats = trainer.run_full_pipeline()

    print()
    print("Training completed!")
    print(f"Training statistics: {trainer_stats}")
    print(f"Model saved to: {export_config.save_local_path}")


if __name__ == "__main__":
    main()

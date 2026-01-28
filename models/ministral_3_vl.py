from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from unsloth import is_bf16_supported
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset as HFDataset
from typing import Dict, List, Any

from finetuning.base_trainer import BaseVisionModelTrainer


class Ministral3VLTrainer(BaseVisionModelTrainer):
    """Trainer for Ministral-3-vl vision-language model."""

    def load_model(self):
        """Load the Ministral-3-vl model and tokenizer."""
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            self.model_config.model_name,
            load_in_4bit=self.model_config.load_in_4bit,
            use_gradient_checkpointing=self.model_config.use_gradient_checkpointing,
        )

    def apply_lora(self):
        """Apply LoRA/PEFT to the Ministral-3-vl model."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model or tokenizer not loaded. Call load_model() first."
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

    def load_dataset(self):
        """Load and prepare the LaTeX OCR dataset."""
        self.dataset = load_dataset(
            self.dataset_config.dataset_name, split=self.dataset_config.dataset_split
        )

        # Convert to conversation format
        converted_dataset = [
            self._convert_to_conversation(sample) for sample in self.dataset
        ]

        # Convert list to Dataset
        self.train_dataset = HFDataset.from_list(converted_dataset)

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
        else:
            # Use max_steps
            max_steps_value = self.training_config.max_steps
            num_train_epochs_value = 1

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

        self.model.push_to_hub_gguf(
            self.export_config.gguf_model_id,
            self.tokenizer,
            quantization_method=self.export_config.gguf_quantization_methods,
            token=token_value,
        )

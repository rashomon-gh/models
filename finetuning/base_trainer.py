from abc import ABC, abstractmethod
from typing import Dict, Optional

from finetuning.config import ModelConfig, DatasetConfig, TrainingConfig, ExportConfig


class BaseVisionModelTrainer(ABC):
    """Abstract base class for vision-language model fine-tuning."""

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
        export_config: Optional[ExportConfig] = None,
    ):
        """
        Initialize the base trainer with configurations.

        Args:
            model_config: Configuration for model loading and LoRA setup
            dataset_config: Configuration for dataset loading and processing
            training_config: Configuration for training parameters
            export_config: Optional configuration for model export
        """
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.training_config = training_config
        self.export_config = export_config or ExportConfig()

        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset = None

    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement load_model()")

    @abstractmethod
    def apply_lora(self):
        """Apply LoRA/PEFT to the model. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement apply_lora()")

    @abstractmethod
    def load_dataset(self):
        """Load and prepare the dataset. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement load_dataset()")

    @abstractmethod
    def prepare_trainer(self):
        """Prepare the trainer. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement prepare_trainer()")

    def train(self) -> Dict[str, float]:
        """
        Train the model.

        Returns:
            Training statistics
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call prepare_trainer() first.")

        return self.trainer.train()

    def export_model(self):
        """Export the model according to export configuration."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not initialized.")

        if self.export_config.save_local_16bit:
            self._save_local()

        if self.export_config.push_to_hub:
            self._push_to_hub()

        if self.export_config.push_gguf:
            self._push_gguf()

    @abstractmethod
    def _save_local(self):
        """Save model locally. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _save_local()")

    @abstractmethod
    def _push_to_hub(self):
        """Push model to Hugging Face Hub. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _push_to_hub()")

    @abstractmethod
    def _push_gguf(self):
        """Push GGUF quantized model to hub. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _push_gguf()")

    def run_full_pipeline(self) -> Dict[str, float]:
        """
        Run the complete training and export pipeline.

        Returns:
            Training statistics
        """
        self.load_model()
        self.apply_lora()
        self.load_dataset()
        self.prepare_trainer()
        stats = self.train()
        self.export_model()

        return stats

import unsloth  # noqa: F401
import wandb
from config.keys import api_keys
from datasets import load_dataset
from loguru import logger
from trl import SFTConfig, SFTTrainer  # type:ignore
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from fire import Fire

logger.info("Setting up Weights and Biases (wandb) for experiment tracking")
wandb.login(key=api_keys.wandb_api_key.get_secret_value())
wandb.init(
    project="Qwen3.5 4B Vision finetune",
    name="Qwen3_5_4B_vision_finetune",
    config={
        "model_name": "unsloth/Qwen3.5-4B",
        "dataset_name": "unsloth/LaTeX_OCR",
        "training_parameters": {
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 5,
            "max_steps": 50,
            "learning_rate": 2e-4,
            "logging_steps": 1,
            "optim": "adamw_8bit",
            "weight_decay": 0.001,
            "lr_scheduler_type": "linear",
        },
        "peft_parameters": {
            "r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0,
            "bias": "none",
            "random_state": 3407,
            "use_rslora": False,
            "loftq_config": None,
            "target_modules": "all-linear",
        },
    },
)


def convert_to_conversation(sample):
    instruction = "Write the LaTeX representation for this image."
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["text"]}]},
    ]
    return {"messages": conversation}


class Qwen3_5_4B_Vision:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.training_dataset = None
        self.validation_dataset = None

        self.model_name = "unsloth/Qwen3.5-4B"
        self.dataset_name = "unsloth/LaTeX_OCR"
        
        logger.info("Initialised empty model object for finetuning.")
        logger.info(f"Model name set to: {self.model_name}")
        logger.info(f"Dataset name set to: {self.dataset_name}")

    def __load_peft_model(self):
        # Load the LoRA adapters from the specified path
        logger.info(f"Loading model: {self.model_name} with LoRA adapters")
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            self.model_name,
            load_in_4bit=False,  # Use 4bit to reduce memory use. False for 16bit LoRA.
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
        )

        # load LoRA adapters
        self.model = FastVisionModel.get_peft_model(
            self.model,
            finetune_vision_layers=True,  # False if not finetuning vision layers
            finetune_language_layers=True,  # False if not finetuning language layers
            finetune_attention_modules=True,  # False if not finetuning attention layers
            finetune_mlp_modules=True,  # False if not finetuning MLP layers
            r=16,  # The larger, the higher the accuracy, but might overfit
            lora_alpha=16,  # Recommended alpha == r at least
            lora_dropout=0,
            bias="none",
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
            target_modules="all-linear",  # Optional now! Can specify a list if needed
        )

    def __load_dataset(self):
        # Load the dataset using Hugging Face's datasets library
        logger.info(f"Loading dataset: {self.dataset_name}")
        self.training_dataset = load_dataset(self.dataset_name, split="train")
        # using the test dataset for validation 
        self.validation_dataset = load_dataset(self.dataset_name, split="test")

    def train(self):
        self.__load_peft_model()
        self.__load_dataset()

        FastVisionModel.for_training(self.model)  # Enable for training!

        logger.info("Initializing SFT Trainer")
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,  # type:ignore
            data_collator=UnslothVisionDataCollator(
                self.model, self.tokenizer
            ),  # Must use!
            train_dataset=self.training_dataset,
            eval_dataset=self.validation_dataset,
            args=SFTConfig(
                per_device_train_batch_size=8,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps = 50,
                # num_train_epochs=50,  # Set this instead of max_steps for full training runs
                learning_rate=2e-4,
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.001,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
                report_to="wandb",  # For Weights and Biases
                # You MUST put the below items for vision finetuning:
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_length=2048,
            ),
        )

        logger.info("Starting training")
        trainer_stats = trainer.train()

        return trainer_stats

    def save_model(self):
        logger.info(f"Saving 16-bit model to {self.model_name}-16bit")
        self.model.save_pretrained_merged( # pyright: ignore[reportOptionalMemberAccess]
            f"{self.model_name}-16bit",
            self.tokenizer,
        )  # type:ignore

    def push_to_hub(self):
        logger.info(f"Pushing model to Hugging Face Hub with name: {self.model_name}")
        self.model.push_to_hub_merged(  # type:ignore
            f"{api_keys.huggingface_username}/{self.model_name}-16bit",
            self.tokenizer,
            token=api_keys.huggingface_token.get_secret_value(),
        )


def main(save: bool = True, push: bool = False):
    model = Qwen3_5_4B_Vision()
    trainer_stats = model.train()

    if save:
        model.save_model()

    if push:
        model.push_to_hub()

    return trainer_stats

if __name__ == "__main__":
    Fire(main)

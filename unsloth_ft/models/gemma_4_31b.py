import torch
import unsloth  # noqa: F401
import wandb
from config.keys import api_keys
from datasets import load_dataset
from logging import setup_logging # type: ignore
from trl import SFTConfig, SFTTrainer  # type:ignore
from unsloth import FastModel
from fire import Fire
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_data_formats
from unsloth.chat_templates import train_on_responses_only


logger = setup_logging()

logger.info("Starting Gemma-4-31B training script")

# TODO: wandb
wandb.login(key=api_keys.wandb_api_key.get_secret_value())
wandb.init(
    project="Gemma-4-31B-SFT", 
    name="Gemma-4-31B-SFT",
    config={
        "model_name": "unsloth/gemma-4-31B-it",
        "dataset_name": "mlabonne/FineTome-100k",
        "training_parameters": {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 5,
            "num_train_epochs": 3,
            "learning_rate": 2e-5,
            "logging_steps": 1,
            "optim": "adamw_8bit",
            "weight_decay": 0.001,
            "lr_scheduler_type": "linear",
        },
        "peft_parameters": {
            "r": 8,
            "lora_alpha": 8,
            "lora_dropout": 0.1,
            "bias": "none",
            "random_state": 3407,
        },
    },
)

class Gemma_4_31B:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.training_dataset = None
        self.validation_dataset = None
        
        self.model_name = "unsloth/gemma-4-31B-it"
        self.dataset_name = "mlabonne/FineTome-100k"
        self.mode_save_name = self.model_name.replace("unsloth/", "")
    
    
    def __load_peft_model(self):
        logger.info(f"Loading model {self.model_name} with PEFT")
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name = self.model_name,
            dtype = torch.bfloat16,
            max_seq_length = 16384,
            load_in_8bit = True, 
            full_finetuning = False, 
            token = api_keys.huggingface_token.get_secret_value(),
        )
        
        self.model = FastModel.get_peft_model(
            self.model,
            finetune_vision_layers     = False, 
            finetune_language_layers   = True,  
            finetune_attention_modules = True,  
            finetune_mlp_modules       = True,  

            r = 8,           
            lora_alpha = 8,
            lora_dropout = 0.1,
            bias = "none",
            random_state = 3407,
        )
        
    def __get_chat_template(self):
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template = "gemma-4-thinking",
        )
        
    def formatting_prompts_func(self, examples):
        convos = examples["conversations"]
        texts = [
            self.tokenizer.apply_chat_template( # type: ignore
                convo, 
                tokenize = False, 
                add_generation_prompt = False
            ).removeprefix('<bos>') for convo in convos] 
        return { "text" : texts, }
        
    def __load_datasets(self):
        logger.info(f"Loading dataset {self.dataset_name}")
        self.training_dataset = load_dataset(self.dataset_name, split="train")
        self.training_dataset = standardize_data_formats(self.training_dataset)
        self.training_dataset = self.training_dataset.map(self.formatting_prompts_func, batched=True)

    def train(self):
        self.__load_peft_model()
        self.__get_chat_template()
        self.__load_datasets()
        
        training_args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4, # Use GA to mimic batch size!
            warmup_steps = 5,
            num_train_epochs = 3,
            learning_rate = 2e-5, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "wandb", # Use TrackIO/WandB etc
        ),
        
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer, # type: ignore
            train_dataset = self.training_dataset,
            eval_dataset = None, # the dataset has no eval split
            args = training_args,
        )
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|turn>user\n",
            response_part="<|turn>model\n",
        )
        
        logger.info("Started Training")
        trainer_stats = trainer.train()
        return trainer_stats
    
    
    def save_model(self):
        logger.info(f"Saving 16-bit model to {self.model_name}-16bit")
        
        save_name = f"{self.model_name.replace('unsloth/', '')}-16bit"
        self.model.save_pretrained_merged( # pyright: ignore[reportOptionalMemberAccess]
            save_name,
            self.tokenizer,
        )  # type:ignore

    def push_to_hub(self):
        save_name = f"{api_keys.huggingface_username}/{self.model_name.replace("unsloth/", "")}-16bit"
        logger.info(f"Pushing model to Hugging Face Hub with name: {self.model_name.replace("unsloth/", "")}")
        self.model.push_to_hub_merged(  # type:ignore
            save_name,
            self.tokenizer,
            token=api_keys.huggingface_token.get_secret_value(),
        )

def main(save: bool = False, push: bool = True):
    model = Gemma_4_31B()
    trainer_stats = model.train()
    logger.info(trainer_stats)

    if save:
        model.save_model()

    if push:
        model.push_to_hub()


if __name__ == "__main__":
    Fire(main)

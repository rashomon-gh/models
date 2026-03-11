import unsloth  # noqa: F401
import wandb
from config.keys import api_keys
from datasets import load_dataset
from loguru import logger
from trl import SFTConfig, SFTTrainer  # type:ignore
from unsloth import FastLanguageModel
from fire import Fire

logger.info("Starting GLM-4.7-Flash training script")
wandb.login(key=api_keys.wandb_api_key.get_secret_value())
wandb.init(
    project="GLM-4.7-Flash-SFT", 
    name="GLM-4.7-Flash-SFT",
    config={
        "model_name": "unsloth/GLM-4.7-Flash",
        "dataset_name": "unsloth/OpenMathReasoning-mini",
        "training_parameters": {
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "warmup_steps": 5,
            "max_steps": 60,
            "learning_rate": 2e-4,
            "logging_steps": 1,
            "optim": "adamw_8bit",
            "weight_decay": 0.001,
            "lr_scheduler_type": "linear",
        },
        "peft_parameters": {
            "r": 128,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj",
                               "out_proj",],
            "lora_alpha": 16,
            "lora_dropout": 0,
            "bias": "none",
            "use_gradient_checkpointing": "unsloth",
            "random_state": 3407,
            "use_rslora": False,
            "loftq_config": None,
        },
    },
)


def generate_conversation(examples):
    problems  = examples["problem"]
    solutions = examples["generated_solution"]
    conversations = []
    for problem, solution in zip(problems, solutions):
        conversations.append([
            {"role" : "user",      "content" : problem},
            {"role" : "assistant", "content" : solution},
        ])
    return { "conversations": conversations, }



class GLM_4_7_Flash:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.training_dataset = None
        self.validation_dataset = None
        
        self.model_name = "unsloth/GLM-4.7-Flash"
        self.dataset_name = "unsloth/OpenMathReasoning-mini"
        self.mode_save_name = self.model_name.replace("unsloth/", "")
        
    def __load_peft_model(self):
        logger.info(f"Loading PEFT model: {self.model_name}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = 64 * 1000, # 64K context length
            load_in_4bit = False,  # 4 bit quantization to reduce memory
            load_in_8bit = False, 
            full_finetuning = False,
            trust_remote_code = True,
            unsloth_force_compile = False,
            token=api_keys.huggingface_token.get_secret_value(),
        )
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",
                            "out_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
        
    def formatting_prompts_func(self, examples):
        convos = examples["conversations"]
        texts = [self.tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos] # type: ignore
        return { "text" : texts, }
        
    
    def __load_dataset(self):
        logger.info(f"Loading dataset: {self.dataset_name}")
        self.training_dataset = load_dataset(self.dataset_name, split="cot")
        logger.info("Generating conversations for training dataset")
        self.training_dataset = self.training_dataset.map(generate_conversation, batched = True)
        logger.info("Formatting prompts for training dataset")
        self.training_dataset = self.training_dataset.map(self.formatting_prompts_func, batched = True)
    
    
    
        
    def train(self):
        self.__load_peft_model()
        self.__load_dataset()
        logger.info("Starting training with SFTTrainer")
        
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer, # type: ignore
            train_dataset = self.training_dataset,
            eval_dataset = None,
            args = SFTConfig(
                dataset_text_field = "text",
                dataset_num_proc = 1, # Increasing "might" throw error on Colab/other envs.
                per_device_train_batch_size = 8,
                gradient_accumulation_steps = 2, # Use GA to mimic batch size!
                warmup_steps = 5,
                # num_train_epochs = 1, # Set this for 1 full training run.
                max_steps = 60,
                learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.001,
                lr_scheduler_type = "linear",
                seed = 3407,
                report_to = "wandb", # Use TrackIO/WandB etc
            )
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
    model = GLM_4_7_Flash()
    trainer_stats = model.train()
    logger.info(trainer_stats)

    if save:
        model.save_model()

    if push:
        model.push_to_hub()


if __name__ == "__main__":
    Fire(main)

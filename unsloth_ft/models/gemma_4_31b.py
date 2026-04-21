import torch
import unsloth  # noqa: F401
import wandb
from config.keys import api_keys
from datasets import load_dataset
from logging import setup_logging
from trl import SFTConfig, SFTTrainer  # type:ignore
from unsloth import FastModel
from fire import Fire
from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_data_formats


logger = setup_logging()

logger.info("Starting Gemma-4-31B training script")

# TODO: wandb


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
        texts = [self.tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
        return { "text" : texts, }
        
    def __load_datasets(self):
        logger.info(f"Loading dataset {self.dataset_name}")
        self.training_dataset = load_dataset(self.dataset_name, split="train")
        self.training_dataset = standardize_data_formats(self.training_dataset)
        self.training_dataset = self.training_dataset.map(self.formatting_prompts_func, batched=True, num_proc=4)

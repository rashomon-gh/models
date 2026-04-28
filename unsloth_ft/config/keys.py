from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger

class ApiKeys(BaseSettings):
    huggingface_token: SecretStr
    huggingface_username: str
    wandb_api_key: SecretStr
    
    model_config = SettingsConfigDict(env_file=".env")


api_keys = ApiKeys() # type: ignore
logger.debug(f"Loaded API keys: {api_keys}")

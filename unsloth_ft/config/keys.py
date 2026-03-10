from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class ApiKeys(BaseSettings):
    huggingface_token: SecretStr
    huggingface_username: str
    wandb_api_key: SecretStr
    
    model_config = SettingsConfigDict(env_file="../.env")


api_keys = ApiKeys() # type: ignore

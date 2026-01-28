from pydantic_settings import BaseSettings
from pydantic import SecretStr


class Settings(BaseSettings):
    huggingface_token: SecretStr

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "forbid"

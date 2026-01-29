from pydantic_settings import BaseSettings
from pydantic import SecretStr


class Settings(BaseSettings):
    huggingface_token: SecretStr
    huggingface_username: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "forbid"


settings = Settings()  # type: ignore[call-arg]

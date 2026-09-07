# pyright: reportCallIssue=false

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """
    Configuration settings for the application.
    """

    # FOR FILE SAVING
    RESULTS_FOLDER: str = ""

    # FOR ADDRESSES STANDARDIZATION
    S2S_MODEL_PATH: str = ""

    # FOR LOGS
    LOGS_PATH: str = ""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()

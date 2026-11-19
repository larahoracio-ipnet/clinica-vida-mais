from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
STORAGE_DIR = BACKEND_DIR / "storage"

class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Clinica Vida+ - Auditoria Inteligente TISS/TUSS"
    app_env: str = "dev"
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    max_upload_mb: int = 10

    databricks_host: str = ""
    databricks_token: str = ""
    databricks_volume_path: str = ""

    @property
    def cors_origins_list(self) -> list[str]: 
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()] 

    @property
    def databricks_configured(self) -> bool:

        return bool(
            self.databricks_host
            and self.databricks_token
            and self.databricks_volume_path
        )
@lru_cache
def get_settings() -> Settings:
    return Settings()
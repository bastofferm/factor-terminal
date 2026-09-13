"""Configuration.

One env var per concern. Deliberately not the DATABASE_URL vs XBRL_SEC_DATABASE_URL
split the source repos carry, which makes it easy to point the API and the batch
jobs at different databases by accident.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    factors_database_url: str = Field(default="postgresql://postgres@127.0.0.1:5432/factors")
    warehouse_database_url: str = Field(default="postgresql://postgres@127.0.0.1:5432/xbrl_sec")
    db_schema: str = Field(default="public")

    fred_api_key: str = Field(default="")

    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8100)
    allowed_origins: str = Field(default="http://localhost:3100,http://127.0.0.1:3100")
    statement_timeout_ms: int = Field(default=120_000)
    pool_min: int = Field(default=2)
    pool_max: int = Field(default=10)
    log_level: str = Field(default="INFO")

    base_ccy: str = Field(default="USD")
    trading_days: int = Field(default=252)
    default_start: str = Field(default="2000-01-03")

    yahoo_rate_limit: float = Field(default=2.0)
    fred_rate_limit: float = Field(default=8.0)
    liveness_stale_days: int = Field(default=10)

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def pg_password(self) -> str:
        """libpq picks PGPASSWORD up by itself for normal connections; we need it
        explicitly only for the FDW user mapping, which stores it server-side."""
        return os.environ.get("PGPASSWORD", "")

    def resolved_fred_key(self) -> str:
        return self.fred_api_key or os.environ.get("FRED_API_KEY", "")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

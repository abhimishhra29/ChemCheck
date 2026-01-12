from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    project_name: str = "ChemCheck"
    api_v1_prefix: str = "/api/v1"


settings = Settings()

"""
Application-wide settings and configuration.

Note: For a production-ready application, it is best practice to use
pydantic-settings.BaseSettings to load configuration from environment
variables, which is more secure and flexible than hardcoding defaults.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """
    A frozen dataclass holding the application's core settings.
    
    Attributes:
        project_name: The name of the application.
        api_v1_prefix: The URL prefix for the v1 API.
    """
    project_name: str = "ChemCheck"
    api_v1_prefix: str = "/api/v1"


settings = Settings()

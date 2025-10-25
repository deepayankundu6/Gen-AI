"""
config.py

Configuration management for the processor application.
Loads all required settings from environment variables and .env file.
"""
import os
from pathlib import Path
from typing import NoReturn

from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Set default values
DEFAULT_CONFIG = {
    "OPENAI_API_BASE": "http://localhost:5000/v1",
    "OPENAI_API_KEY": "Dummy api key",
    "MONGO_URI": "mongodb://localhost:27017",
    "DB_NAME": "Gen_AI",
    "EMBEDDING_COLLECTION": "Embedings",
    "RESULT_COLLECTION": "Result",
    "EMBEDDING_MODEL": "text-embedding-3-small",
    "RESPONSE_MODEL": "deepseek/deepseek-r1-0528-qwen3-8b",
    "BATCH_SIZE": "16"
}


def get_env_var(name: str) -> str:
    """Get an environment variable or use default if not set."""
    value = os.getenv(name)
    if value is None:
        if name in DEFAULT_CONFIG:
            value = DEFAULT_CONFIG[name]
            # Set it in environment for consistency
            os.environ[name] = value
        else:
            raise ValueError(f"Environment variable {name} must be set")
    return value


# API Configuration
def get_api_config() -> tuple[str, str]:
    """Get API configuration from environment."""
    api_base = get_env_var("OPENAI_API_BASE")
    api_key = get_env_var("OPENAI_API_KEY")
    return api_base, api_key


# MongoDB Configuration
def get_mongo_config() -> tuple[str, str, str, str]:
    """Get MongoDB configuration from environment."""
    uri = get_env_var("MONGO_URI")
    db_name = get_env_var("DB_NAME")
    embedding_collection = get_env_var("EMBEDDING_COLLECTION")
    result_collection = get_env_var("RESULT_COLLECTION")
    return uri, db_name, embedding_collection, result_collection


# Model Configuration
def get_model_config() -> tuple[str, str]:
    """Get model configuration from environment."""
    embedding_model = get_env_var("EMBEDDING_MODEL")
    response_model = get_env_var("RESPONSE_MODEL")
    return embedding_model, response_model


# Processing Configuration
def get_batch_size() -> int:
    """Get batch size from environment."""
    try:
        return int(get_env_var("BATCH_SIZE"))
    except ValueError as e:
        raise ValueError("BATCH_SIZE must be a valid integer") from e


def validate_url(url: str, name: str) -> None:
    """Validate URL format."""
    if not url.startswith(('http://', 'https://')):
        raise ValueError(f"{name} must start with http:// or https://")


def validate_mongo_uri(uri: str) -> None:
    """Validate MongoDB URI format."""
    if not uri.startswith(('mongodb://', 'mongodb+srv://')):
        raise ValueError("MONGO_URI must start with mongodb:// or mongodb+srv://")


def validate_config() -> None:
    """Validate all required environment variables are set and valid."""
    required_vars = [
        "OPENAI_API_BASE",
        "OPENAI_API_KEY",
        "MONGO_URI",
        "DB_NAME",
        "EMBEDDING_COLLECTION",
        "RESULT_COLLECTION",
        "EMBEDDING_MODEL",
        "RESPONSE_MODEL",
        "BATCH_SIZE"
    ]
    
    # First ensure all variables have values (either from env or defaults)
    for var in required_vars:
        get_env_var(var)
    
    # Validate URL formats
    validate_url(get_env_var("OPENAI_API_BASE"), "OPENAI_API_BASE")
    validate_mongo_uri(get_env_var("MONGO_URI"))
    
    # Validate batch size is integer
    try:
        batch_size = int(get_env_var("BATCH_SIZE"))
        if batch_size <= 0:
            raise ValueError
    except ValueError:
        raise ValueError("BATCH_SIZE must be a positive integer")
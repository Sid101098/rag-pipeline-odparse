import yaml
from typing import Dict, Any
import logging
import os
from dotenv import load_dotenv

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_environment_variables():
    """Load environment variables from .env file"""
    load_dotenv()
    
    # Validate required environment variables
    required_vars = ['OPENAI_API_KEY']
    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"Missing required environment variable: {var}")

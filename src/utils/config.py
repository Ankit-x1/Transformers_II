import yaml 
import os
from pathlib import Path 
from typing import Dict, Any
import hashlib 
import json

class Config:
    """
    Configuration Management system
    """

    def __init__(self, config_path:str):
        self.config_path = config_path
        self._config = self._load_config()
        self._validate_config()
        self.config_hash = self._compute_hash()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration"""

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config 
    
    def _validate_config(self):
        """Validate required fields exist"""

        required_sections = ['project', 'data', 'model', 'training']
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required config section: {section}")

    def _compute_hash(self) -> str:
        """
        Computer deterministic hash of config for experiment tracking
        """     

        config_str = json.dumps(self._config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    

    def get(self, *keys, default=None):
        """
        Get nested config value.
        Example: config.get('model', 'transformer', 'd_model')
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value 
    
    def __get_item__(self, key):
        return self._config[key]
    
    def __repr__(self):
        return f"Config(hash={self.config_hash}, path={self.config_path})"
    
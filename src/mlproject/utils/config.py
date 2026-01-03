"""Configuration loading utilities"""

from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import Union
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        OmegaConf configuration object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    config = OmegaConf.load(config_path)
    
    return config


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple configuration objects.
    
    Args:
        *configs: Variable number of configuration objects
        
    Returns:
        Merged configuration
    """
    return OmegaConf.merge(*configs)


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        OmegaConf.save(config, f)
    
    logger.info(f"Saved configuration to {output_path}")

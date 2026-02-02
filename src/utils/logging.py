import logging 
import sys 
from pathlib import Path 
from datetime import datetime 
from typing import Optional 

class MLOpsLogger:
    """Production logging system.
    
    Features:
    - Dual output
    - Timestamp 
    - Different levels for prodcution Components
    - Structured logging for metrics
    """
    
    def __init__(
            self, 
            name:str,
            log_dir: str = "logs",
            level: int = logging.INFO,
            log_to_file: bool = True):
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        if log_to_file:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = Path(log_dir) / f"{name}_{timestamp}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, msg:str):
        self.logger.info(msg)

    def warning(self, msg:str):
        self.logger.warning(msg)
    
    def error(self, msg:str):
        self.logger.error(msg)
    
    def debug(self, msg:str):
        self.logger.debug(msg)

    
    def log_metrics(self, metrics:dict, step: Optional[int] = None):
        """
        Log metrics in structured Format
        """
        step_str = f"[Step {step}] " if step is not None else ""
        metrics_str = " | ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                  for k, v in metrics.items()])
        self.logger.info(f"{step_str}{metrics_str}")
    
    def log_config(self, config:dict):
        """Log configuration at start of experiment
        """
        self.logger.info('=' * 80)
        self.logger.info("Experiment Configuration")
        self.logger.info('=' * 80)

        for key, value in config.items():
            if isinstance(value, dict):
                self.logger.info(f"{key}:")
                for k,v in value.items():
                    self.logger.info(f" {k}: {v}")
            else:
                self.logger.info(f"{key}: {value}")

"""
Main Steering script
"""

import logger
log = logger.get_logger(__name__)
import hydra
from omegaconf import DictConfig
import traceback
import sys
import run

@hydra.main(config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    
    run.simple_1D_training(config)
    
    
if __name__ == "__main__":
    # Run selected script. If arg invaild exit
    try:
        main()
        sys.exit(0)
    except Exception:
        log.fatal(traceback.format_exc())
        sys.exit(1)
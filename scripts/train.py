"""Standalone training script"""

import sys
import logging
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlproject.utils.logging_config import setup_logging
from mlproject.utils.config import load_config
import pandas as pd
from mlproject.train.trainer import ModelTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train NYC Taxi duration prediction models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to features data (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save models (overrides config)"
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        help="Directory to save metrics (overrides config)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    return parser.parse_args()


def main():
    """Run model training."""
    args = parse_args()
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Running training script...")
    
    try:
        config = load_config(args.config)
        
        # Load features
        input_path = args.input or f"{config.data.features_dir}/train_features.csv"
        features_df = pd.read_csv(input_path, parse_dates=['pickup_datetime', 'dropoff_datetime'])
        
        # Train
        trainer = ModelTrainer({
            'test_size': config.train.test_size,
            'random_state': config.train.random_state,
            'metric': config.train.metric,
            'models': config.model.models,
            'output_dir': args.output_dir or config.model.output_dir,
            'metrics_dir': args.metrics_dir or config.model.metrics_dir
        })
        
        results = trainer.train(features_df)
        
        logger.info(f"Training complete! Best model: {results['best_model']}")
        logger.info(f"Models saved to: {args.output_dir or config.model.output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

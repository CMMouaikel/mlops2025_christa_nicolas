"""Standalone preprocessing script"""

import sys
import logging
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlproject.utils.logging_config import setup_logging
from mlproject.utils.config import load_config
from mlproject.utils.loader import load_train_data, save_data
from mlproject.preprocess.cleaner import DataPreprocessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess NYC Taxi data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input training data (overrides config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save cleaned data (overrides config)"
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
    """Run data preprocessing."""
    args = parse_args()
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Running data preprocessing script...")
    
    try:
        config = load_config(args.config)
        
        # Load data
        input_path = args.input or config.data.train_path
        train_df = load_train_data(input_path)
        
        # Preprocess
        preprocessor = DataPreprocessor(config.preprocessing)
        clean_df = preprocessor.fit_transform(train_df, is_train=True)
        
        # Save
        if args.output:
            output_path = args.output
        else:
            output_path = f"{config.data.processed_dir}/train_cleaned.csv"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_data(clean_df, output_path)
        
        logger.info(f"Preprocessing complete! Saved to: {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

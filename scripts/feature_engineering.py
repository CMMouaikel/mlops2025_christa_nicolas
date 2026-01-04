"""Standalone feature engineering script"""

import sys
import logging
import argparse
import pickle
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlproject.utils.logging_config import setup_logging
from mlproject.utils.config import load_config
from mlproject.utils.loader import save_data
import pandas as pd
from mlproject.features.engineer import FeatureEngineer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Engineer features for NYC Taxi data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to cleaned data (overrides config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save features (overrides config)"
    )
    parser.add_argument(
        "--save-engineer",
        type=str,
        help="Path to save fitted feature engineer as pickle"
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
    """Run feature engineering."""
    args = parse_args()
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Running feature engineering script...")
    
    try:
        config = load_config(args.config)
        
        # Load cleaned data
        input_path = args.input or f"{config.data.processed_dir}/train_cleaned.csv"
        clean_df = pd.read_csv(input_path, parse_dates=['pickup_datetime', 'dropoff_datetime'])
        
        # Engineer features
        feature_engineer = FeatureEngineer(config.features)
        features_df = feature_engineer.fit_transform(clean_df, is_train=True)
        
        # Save features
        if args.output:
            output_path = args.output
        else:
            output_path = f"{config.data.features_dir}/train_features.csv"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_data(features_df, output_path)
        logger.info(f"Features saved to: {output_path}")
        
        # Save fitted feature engineer
        if args.save_engineer:
            engineer_path = args.save_engineer
        else:
            engineer_path = f"{config.data.features_dir}/feature_engineer.pkl"
        
        Path(engineer_path).parent.mkdir(parents=True, exist_ok=True)
        with open(engineer_path, 'wb') as f:
            pickle.dump(feature_engineer, f)
        logger.info(f"Feature engineer saved to: {engineer_path}")
        
        logger.info("Feature engineering complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

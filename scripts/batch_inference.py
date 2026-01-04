"""Standalone batch inference script"""

import sys
import logging
import argparse
import pickle
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlproject.utils.logging_config import setup_logging
from mlproject.utils.config import load_config
from mlproject.utils.loader import load_test_data
from mlproject.preprocess.cleaner import DataPreprocessor
from mlproject.features.engineer import FeatureEngineer
from mlproject.train.trainer import load_model
from mlproject.inference.predictor import BatchInferenceEngine


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run batch inference for NYC Taxi predictions")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to test data (overrides config)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model (overrides config)"
    )
    parser.add_argument(
        "--feature-engineer",
        type=str,
        help="Path to fitted feature engineer pickle file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save predictions (overrides config)"
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
    """Run batch inference."""
    args = parse_args()
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Running batch inference script...")
    
    try:
        config = load_config(args.config)
        
        # Load test data
        input_path = args.input or config.data.test_path
        test_df = load_test_data(input_path)
        
        # Preprocess
        preprocessor = DataPreprocessor(config.preprocessing)
        clean_df = preprocessor.transform(test_df, is_train=False)
        
        # Engineer features
        if args.feature_engineer:
            logger.info(f"Loading feature engineer from: {args.feature_engineer}")
            with open(args.feature_engineer, 'rb') as f:
                feature_engineer = pickle.load(f)
        else:
            engineer_path = f"{config.data.features_dir}/feature_engineer.pkl"
            if Path(engineer_path).exists():
                logger.info(f"Loading feature engineer from: {engineer_path}")
                with open(engineer_path, 'rb') as f:
                    feature_engineer = pickle.load(f)
            else:
                logger.info("Creating new feature engineer from config")
                feature_engineer = FeatureEngineer(config.features)
        
        features_df = feature_engineer.transform(clean_df, is_train=False)
        
        # Load model and predict
        if args.model:
            model_path = args.model
        else:
            model_path = Path(config.model.output_dir) / "best_model.pkl"
        
        logger.info(f"Loading model from: {model_path}")
        model = load_model(str(model_path))
        
        inference_engine = BatchInferenceEngine(
            model=model,
            output_dir=args.output or config.inference.output_dir
        )
        
        output_file = inference_engine.predict_and_save(features_df)
        logger.info(f"Predictions saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

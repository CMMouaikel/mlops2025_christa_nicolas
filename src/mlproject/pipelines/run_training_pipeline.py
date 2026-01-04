"""Training Pipeline Orchestrator

Orchestrates the complete training pipeline with support for:
- Local execution
- Docker execution  
- SageMaker Pipeline execution

Environment is auto-detected or can be explicitly specified.
"""

import sys
import os
import argparse
import logging
import pickle
from pathlib import Path

from mlproject.utils.logging_config import setup_logging
from mlproject.utils.config import load_config
from mlproject.utils.loader import load_train_data, save_data
from mlproject.preprocess.cleaner import DataPreprocessor
from mlproject.features.engineer import FeatureEngineer
from mlproject.train.trainer import ModelTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the complete training pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--sagemaker-config",
        type=str,
        default="configs/sagemaker.yaml",
        help="Path to SageMaker configuration file (SageMaker mode only)"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to training data (overrides config). For SageMaker: S3 path"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="auto",
        choices=["auto", "local", "docker", "sagemaker"],
        help="Execution environment: auto-detect, local, docker, or sagemaker"
    )
    parser.add_argument(
        "--pipeline-name",
        type=str,
        help="Override pipeline name (SageMaker only)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for pipeline execution to complete (SageMaker only)"
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Only create/update pipeline, don't execute (SageMaker only)"
    )
    return parser.parse_args()


def detect_environment():
    """Auto-detect execution environment."""
    # Check for SageMaker environment variables
    if os.getenv('SM_TRAINING_ENV') or os.getenv('SM_CURRENT_HOST'):
        return 'sagemaker'
    # Check if running in Docker
    elif os.path.exists('/.dockerenv'):
        return 'docker'
    else:
        return 'local'


def run_local_training(config, args, logger):
    """Run training pipeline locally or in Docker."""
    logger.info("Running training pipeline in LOCAL/DOCKER mode")
    
    # Step 1: Load training data
    logger.info("\n[Step 1/4] Loading training data...")
    input_path = args.input or config.data.train_path
    logger.info(f"Reading from: {input_path}")
    train_df = load_train_data(input_path)
    
    # Step 2: Preprocessing
    logger.info("\n[Step 2/4] Preprocessing data...")
    preprocessor = DataPreprocessor(config.preprocessing)
    clean_df = preprocessor.transform(train_df, is_train=True)
    
    # Save cleaned data
    processed_dir = Path(config.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / "train_cleaned.csv"
    save_data(clean_df, processed_path)
    logger.info(f"Cleaned data saved to: {processed_path}")
    
    # Step 3: Feature engineering
    logger.info("\n[Step 3/4] Engineering features...")
    feature_engineer = FeatureEngineer(config.features)
    features_df = feature_engineer.fit_transform(clean_df, is_train=True)
    
    # Save features and feature engineer
    features_dir = Path(config.data.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    features_path = features_dir / "train_features.csv"
    save_data(features_df, features_path)
    logger.info(f"Features saved to: {features_path}")
    
    # Save fitted feature engineer
    engineer_path = features_dir / "feature_engineer.pkl"
    with open(engineer_path, 'wb') as f:
        pickle.dump(feature_engineer, f)
    logger.info(f"Feature engineer saved to: {engineer_path}")
    
    # Step 4: Model training
    logger.info("\n[Step 4/4] Training models...")
    trainer = ModelTrainer(config.model)
    results = trainer.train(features_df)
    
    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info(f"Best model: {results['best_model_name']}")
    logger.info(f"Best score (RMSE): {results['best_score']:.4f}")
    logger.info(f"Model saved to: {results['model_path']}")
    logger.info("=" * 80)
    
    return 0


def run_sagemaker_training(config, args, logger):
    """Run training pipeline on SageMaker."""
    logger.info("Running training pipeline in SAGEMAKER mode")
    
    try:
        # Import SageMaker dependencies
        try:
            import boto3
            import sagemaker
            from sagemaker.workflow.pipeline import Pipeline
            from sagemaker.workflow.steps import ProcessingStep, TrainingStep
            from sagemaker.workflow.parameters import ParameterString
            from sagemaker.sklearn.processing import SKLearnProcessor
            from sagemaker.sklearn.estimator import SKLearn
            from sagemaker.processing import ProcessingInput, ProcessingOutput
        except ImportError:
            logger.error("Required SageMaker packages not installed")
            logger.error("Install with: uv add boto3 sagemaker")
            raise
        
        # Load SageMaker configuration
        sm_config = load_config(args.sagemaker_config)
        
        # Initialize SageMaker session
        sagemaker_session = sagemaker.Session()
        role = sm_config.sagemaker.role
        bucket = sm_config.sagemaker.bucket
        region = sm_config.sagemaker.region
        
        logger.info(f"Region: {region}")
        logger.info(f"S3 Bucket: {bucket}")
        logger.info(f"IAM Role: {role}")
        
        # Pipeline parameters
        input_data_param = ParameterString(
            name="InputData",
            default_value=args.input or f"s3://{bucket}/nyc-taxi/train.csv"
        )
        
        logger.info(f"\nInput data: {input_data_param.default_value}")
        
        # ============================================================
        # Step 1: Preprocessing
        # ============================================================
        logger.info("\n[Pipeline Step 1/3] Configuring Preprocessing...")
        
        sklearn_processor = SKLearnProcessor(
            framework_version="1.2-1",
            role=role,
            instance_type=sm_config.sagemaker.instance_type,
            instance_count=1,
            base_job_name="nyc-taxi-preprocess",
            sagemaker_session=sagemaker_session,
        )
        
        preprocessing_step = ProcessingStep(
            name="Preprocessing",
            processor=sklearn_processor,
            code="scripts/preprocess.py",
            job_arguments=[
                "--input", "/opt/ml/processing/input/train.csv",
                "--output", "/opt/ml/processing/output/train_cleaned.csv",
                "--config", "configs/train.yaml"
            ],
            inputs=[
                ProcessingInput(
                    source=input_data_param,
                    destination="/opt/ml/processing/input"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="cleaned_data",
                    source="/opt/ml/processing/output",
                    destination=f"s3://{bucket}/nyc-taxi/processed"
                )
            ],
        )
        
        # ============================================================
        # Step 2: Feature Engineering
        # ============================================================
        logger.info("[Pipeline Step 2/3] Configuring Feature Engineering...")
        
        feature_step = ProcessingStep(
            name="FeatureEngineering",
            processor=sklearn_processor,
            code="scripts/feature_engineering.py",
            job_arguments=[
                "--input", "/opt/ml/processing/input/train_cleaned.csv",
                "--output", "/opt/ml/processing/output/train_features.csv",
                "--save-engineer", "/opt/ml/processing/output/feature_engineer.pkl",
                "--config", "configs/train.yaml"
            ],
            inputs=[
                ProcessingInput(
                    source=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                        "cleaned_data"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/input"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="features",
                    source="/opt/ml/processing/output",
                    destination=f"s3://{bucket}/nyc-taxi/features"
                )
            ],
        )
        
        # ============================================================
        # Step 3: Model Training
        # ============================================================
        logger.info("[Pipeline Step 3/3] Configuring Model Training...")
        
        sklearn_estimator = SKLearn(
            entry_point="scripts/train.py",
            role=role,
            instance_type=sm_config.sagemaker.instance_type,
            framework_version="1.2-1",
            py_version="py3",
            hyperparameters={
                "config": "configs/train.yaml"
            },
            sagemaker_session=sagemaker_session,
        )
        
        training_step = TrainingStep(
            name="ModelTraining",
            estimator=sklearn_estimator,
            inputs={
                "train": sagemaker.inputs.TrainingInput(
                    s3_data=feature_step.properties.ProcessingOutputConfig.Outputs[
                        "features"
                    ].S3Output.S3Uri,
                    content_type="text/csv"
                )
            }
        )
        
        # ============================================================
        # Create Pipeline
        # ============================================================
        pipeline_name = args.pipeline_name or sm_config.pipeline.training.name
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Creating Pipeline: {pipeline_name}")
        logger.info(f"{'='*80}")
        
        pipeline = Pipeline(
            name=pipeline_name,
            parameters=[input_data_param],
            steps=[preprocessing_step, feature_step, training_step],
            sagemaker_session=sagemaker_session,
        )
        
        # Upsert pipeline (create or update)
        logger.info("Upserting pipeline definition...")
        pipeline.upsert(role_arn=role)
        logger.info(f"✓ Pipeline '{pipeline_name}' created/updated successfully")
        
        # Execute pipeline
        if not args.no_execute:
            logger.info(f"\n{'='*80}")
            logger.info("Starting Pipeline Execution")
            logger.info(f"{'='*80}")
            
            execution = pipeline.start()
            execution_arn = execution.arn
            
            logger.info(f"✓ Pipeline execution started")
            logger.info(f"Execution ARN: {execution_arn}")
            
            # Wait for completion if requested
            if args.wait:
                logger.info("\nWaiting for pipeline execution to complete...")
                logger.info("(This may take several minutes)")
                execution.wait()
                
                status = execution.describe()['PipelineExecutionStatus']
                logger.info(f"\n{'='*80}")
                logger.info(f"Pipeline Execution Status: {status}")
                logger.info(f"{'='*80}")
                
                if status == "Succeeded":
                    logger.info("✓ Training pipeline completed successfully!")
                    logger.info(f"\nModel artifacts saved to S3")
                    return 0
                else:
                    logger.error(f"✗ Pipeline execution failed with status: {status}")
                    return 1
            else:
                logger.info(f"\n{'='*80}")
                logger.info("Pipeline execution started in background")
                logger.info(f"{'='*80}")
                logger.info("\nTo monitor execution:")
                logger.info(f"  • AWS Console: https://console.aws.amazon.com/sagemaker/home?region={region}#/pipelines/{pipeline_name}")
                logger.info(f"  • AWS CLI: aws sagemaker describe-pipeline-execution --pipeline-execution-arn {execution_arn}")
                logger.info(f"\nTo wait for completion, run with --wait flag")
                return 0
        else:
            logger.info(f"\n{'='*80}")
            logger.info("Pipeline created/updated (not executed)")
            logger.info(f"{'='*80}")
            logger.info(f"\nTo execute pipeline:")
            logger.info(f"  • AWS Console: https://console.aws.amazon.com/sagemaker/home?region={region}#/pipelines/{pipeline_name}")
            logger.info(f"  • Or run this script without --no-execute flag")
            return 0
            
    except Exception as e:
        logger.error(f"\n✗ SageMaker pipeline creation/execution failed: {e}", exc_info=True)
        return 1


def main():
    """Orchestrate training pipeline based on environment."""
    args = parse_args()
    
    # Auto-detect environment if needed
    if args.env == "auto":
        args.env = detect_environment()
    
    # Setup logging
    setup_logging(log_level=args.log_level, log_file="outputs/training.log")
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info(f"Training Pipeline Orchestrator [Environment: {args.env.upper()}]")
    logger.info("=" * 80)
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Configuration loaded from: {args.config}")
        
        # Route to appropriate execution mode
        if args.env in ["local", "docker"]:
            return run_local_training(config, args, logger)
        elif args.env == "sagemaker":
            return run_sagemaker_training(config, args, logger)
        else:
            logger.error(f"Unknown environment: {args.env}")
            return 1
            
    except Exception as e:
        logger.error(f"\n✗ Training pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

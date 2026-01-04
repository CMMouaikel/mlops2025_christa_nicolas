"""Batch Inference Pipeline Orchestrator

Orchestrates the complete batch inference pipeline with support for:
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
from mlproject.utils.loader import load_test_data
from mlproject.preprocess.cleaner import DataPreprocessor
from mlproject.features.engineer import FeatureEngineer
from mlproject.train.trainer import load_model
from mlproject.inference.predictor import BatchInferenceEngine


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the complete batch inference pipeline")
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
        help="Path to test data (overrides config). For SageMaker: S3 path"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model (overrides config). For SageMaker: S3 path"
    )
    parser.add_argument(
        "--feature-engineer",
        type=str,
        help="Path to feature engineer pickle file. For SageMaker: S3 path"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save predictions (overrides config). For SageMaker: S3 path"
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


def run_local_inference(config, args, logger):
    """Run batch inference pipeline locally or in Docker."""
    logger.info("Running inference pipeline in LOCAL/DOCKER mode")
    
    # Step 1: Load test data
    logger.info("\n[Step 1/4] Loading test data...")
    input_path = args.input or config.data.test_path
    logger.info(f"Reading from: {input_path}")
    test_df = load_test_data(input_path)
    
    # Step 2: Preprocessing
    logger.info("\n[Step 2/4] Preprocessing data...")
    preprocessor = DataPreprocessor(config.preprocessing)
    clean_df = preprocessor.transform(test_df, is_train=False)
    
    # Step 3: Feature engineering
    logger.info("\n[Step 3/4] Engineering features...")
    
    # Try to load fitted feature engineer
    if args.feature_engineer:
        engineer_path = args.feature_engineer
    else:
        engineer_path = f"{config.data.features_dir}/feature_engineer.pkl"
    
    if Path(engineer_path).exists():
        logger.info(f"Loading fitted feature engineer from: {engineer_path}")
        with open(engineer_path, 'rb') as f:
            feature_engineer = pickle.load(f)
    else:
        logger.warning(f"Feature engineer not found at {engineer_path}")
        logger.warning("Creating new feature engineer from config (may cause inconsistency!)")
        feature_engineer = FeatureEngineer(config.features)
    
    features_df = feature_engineer.transform(clean_df, is_train=False)
    
    # Step 4: Load model and run inference
    logger.info("\n[Step 4/4] Running batch inference...")
    
    if args.model:
        model_path = args.model
    else:
        model_path = Path(config.model.output_dir) / "best_model.pkl"
    
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please run training pipeline first: uv run train"
        )
    
    logger.info(f"Loading model from: {model_path}")
    model = load_model(str(model_path))
    
    output_dir = args.output or config.inference.output_dir
    inference_engine = BatchInferenceEngine(
        model=model,
        output_dir=output_dir
    )
    
    output_file = inference_engine.predict_and_save(features_df)
    
    logger.info("=" * 80)
    logger.info(f"Inference complete!")
    logger.info(f"Predictions saved to: {output_file}")
    logger.info("=" * 80)
    
    return 0


def run_sagemaker_inference(config, args, logger):
    """Run batch inference pipeline on SageMaker."""
    logger.info("Running inference pipeline in SAGEMAKER mode")
    
    try:
        # Import SageMaker dependencies
        try:
            import boto3
            import sagemaker
            from sagemaker.workflow.pipeline import Pipeline
            from sagemaker.workflow.steps import ProcessingStep, TransformStep
            from sagemaker.workflow.parameters import ParameterString
            from sagemaker.sklearn.processing import SKLearnProcessor
            from sagemaker.processing import ProcessingInput, ProcessingOutput
            from sagemaker.transformer import Transformer
            from sagemaker.sklearn.model import SKLearnModel
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
            default_value=args.input or f"s3://{bucket}/nyc-taxi/test.csv"
        )
        
        model_path_param = ParameterString(
            name="ModelPath",
            default_value=args.model or f"s3://{bucket}/nyc-taxi/models"
        )
        
        engineer_path_param = ParameterString(
            name="FeatureEngineerPath",
            default_value=args.feature_engineer or f"s3://{bucket}/nyc-taxi/artifacts/feature_engineer.pkl"
        )
        
        output_path_param = ParameterString(
            name="OutputPath",
            default_value=args.output or f"s3://{bucket}/nyc-taxi/predictions"
        )
        
        logger.info(f"\nInput data: {input_data_param.default_value}")
        logger.info(f"Model path: {model_path_param.default_value}")
        logger.info(f"Feature engineer: {engineer_path_param.default_value}")
        logger.info(f"Output path: {output_path_param.default_value}")
        
        # ============================================================
        # Step 1: Preprocessing
        # ============================================================
        logger.info("\n[Pipeline Step 1/3] Configuring Preprocessing...")
        
        sklearn_processor = SKLearnProcessor(
            framework_version="1.2-1",
            role=role,
            instance_type=sm_config.sagemaker.instance_type,
            instance_count=1,
            base_job_name="nyc-taxi-inference-preprocess",
            sagemaker_session=sagemaker_session,
        )
        
        preprocessing_step = ProcessingStep(
            name="InferencePreprocessing",
            processor=sklearn_processor,
            code="scripts/preprocess.py",
            job_arguments=[
                "--input", "/opt/ml/processing/input/test.csv",
                "--output", "/opt/ml/processing/output/test_cleaned.csv",
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
                    destination=f"s3://{bucket}/nyc-taxi/inference/processed"
                )
            ],
        )
        
        # ============================================================
        # Step 2: Feature Engineering
        # ============================================================
        logger.info("[Pipeline Step 2/3] Configuring Feature Engineering...")
        
        feature_step = ProcessingStep(
            name="InferenceFeatureEngineering",
            processor=sklearn_processor,
            code="scripts/feature_engineering.py",
            job_arguments=[
                "--input", "/opt/ml/processing/input/test_cleaned.csv",
                "--output", "/opt/ml/processing/output/test_features.csv",
                "--feature-engineer", "/opt/ml/processing/engineer/feature_engineer.pkl",
                "--config", "configs/train.yaml"
            ],
            inputs=[
                ProcessingInput(
                    source=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                        "cleaned_data"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/input"
                ),
                ProcessingInput(
                    source=engineer_path_param,
                    destination="/opt/ml/processing/engineer"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="features",
                    source="/opt/ml/processing/output",
                    destination=f"s3://{bucket}/nyc-taxi/inference/features"
                )
            ],
        )
        
        # ============================================================
        # Step 3: Batch Transform (Inference)
        # ============================================================
        logger.info("[Pipeline Step 3/3] Configuring Batch Transform...")
        
        # Create SKLearn Model
        sklearn_model = SKLearnModel(
            model_data=model_path_param,
            role=role,
            entry_point="scripts/batch_inference.py",
            framework_version="1.2-1",
            py_version="py3",
            sagemaker_session=sagemaker_session,
        )
        
        # Create Transformer
        transformer = Transformer(
            model_name=sklearn_model.name,
            instance_count=1,
            instance_type=sm_config.sagemaker.instance_type,
            output_path=output_path_param,
            sagemaker_session=sagemaker_session,
        )
        
        transform_step = TransformStep(
            name="BatchInference",
            transformer=transformer,
            inputs=sagemaker.inputs.TransformInput(
                data=feature_step.properties.ProcessingOutputConfig.Outputs[
                    "features"
                ].S3Output.S3Uri,
                content_type="text/csv"
            )
        )
        
        # ============================================================
        # Create Pipeline
        # ============================================================
        pipeline_name = args.pipeline_name or sm_config.pipeline.inference.name
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Creating Pipeline: {pipeline_name}")
        logger.info(f"{'='*80}")
        
        pipeline = Pipeline(
            name=pipeline_name,
            parameters=[input_data_param, model_path_param, engineer_path_param, output_path_param],
            steps=[preprocessing_step, feature_step, transform_step],
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
                    logger.info("✓ Batch inference pipeline completed successfully!")
                    logger.info(f"\nPredictions saved to: {output_path_param.default_value}")
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
    """Orchestrate batch inference pipeline based on environment."""
    args = parse_args()
    
    # Auto-detect environment if needed
    if args.env == "auto":
        args.env = detect_environment()
    
    # Setup logging
    setup_logging(log_level=args.log_level, log_file="outputs/inference.log")
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info(f"Batch Inference Pipeline Orchestrator [Environment: {args.env.upper()}]")
    logger.info("=" * 80)
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Configuration loaded from: {args.config}")
        
        # Route to appropriate execution mode
        if args.env in ["local", "docker"]:
            return run_local_inference(config, args, logger)
        elif args.env == "sagemaker":
            return run_sagemaker_inference(config, args, logger)
        else:
            logger.error(f"Unknown environment: {args.env}")
            return 1
            
    except Exception as e:
        logger.error(f"\n✗ Inference pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

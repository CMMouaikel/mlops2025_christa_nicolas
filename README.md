# NYC Taxi Trip Duration Prediction - MLOps Project

A production-ready machine learning pipeline for predicting NYC taxi trip durations using MLOps best practices.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Model & Metrics](#model--metrics)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Running Locally](#running-locally)
- [Running with Docker](#running-with-docker)
- [Running on AWS SageMaker](#running-on-aws-sagemaker)
- [Team Contributions](#team-contributions)

## Project Overview

This project implements an end-to-end MLOps pipeline for predicting taxi trip durations in New York City. The system processes raw taxi data, engineers relevant features, trains multiple models, and provides batch inference capabilities with support for local, Docker, and AWS SageMaker deployments.

**Key Features:**
- Automated data preprocessing and feature engineering
- Multi-model training with hyperparameter optimization
- Batch inference pipeline
- Docker containerization
- AWS SageMaker integration
- Comprehensive testing suite

## Project Structure

```text
mlops2025_christa_nicolas/
├── configs/
│   ├── train.yaml              # Training configuration
│   └── sagemaker.yaml          # SageMaker-specific config
├── src/
│   └── mlproject/
│       ├── preprocess/
│       │   ├── __init__.py
│       │   └── cleaner.py      # Data cleaning & preprocessing
│       ├── features/
│       │   ├── __init__.py
│       │   └── engineer.py     # Feature engineering
│       ├── train/
│       │   ├── __init__.py
│       │   └── trainer.py      # Model training logic
│       ├── inference/
│       │   ├── __init__.py
│       │   └── predictor.py    # Batch inference engine
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── run_training_pipeline.py      # Training orchestrator
│       │   └── run_batch_inference_pipeline.py
│       └── utils/
│           ├── config.py       # Configuration management
│           ├── loader.py       # Data loading utilities
│           └── logging_config.py
├── scripts/
│   ├── preprocess.py           # Standalone preprocessing
│   ├── feature_engineering.py  # Standalone feature engineering
│   ├── train.py                # Standalone training
│   └── batch_inference.py      # Standalone inference
├── tests/
│   └── test_preprocess.py      # Unit tests
├── notebooks/                  # Exploratory analysis
├── outputs/
│   ├── models/                 # Trained models
│   └── metrics/                # Training metrics
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Multi-container orchestration
├── pyproject.toml              # Project dependencies
└── README.md
```


### Directory Breakdown

- **`configs/`**: YAML configuration files for different environments
- **`src/mlproject/`**: Core Python package with modular components
- **`scripts/`**: Standalone scripts for individual pipeline steps
- **`tests/`**: Unit and integration tests
- **`outputs/`**: Generated artifacts (models, metrics, predictions)

## Model & Metrics

### Selected Metrics

**Primary Metric: RMSE (Root Mean Squared Error)**
- **Justification**: RMSE heavily penalizes large prediction errors, which is critical for taxi duration prediction where significantly overestimating trip time could lead to customer dissatisfaction and underestimating could cause logistical issues.
- **Secondary Metrics**: 
  - MAE (Mean Absolute Error): Provides interpretable average error in seconds
  - R² Score: Indicates overall model fit quality
  - RMSLE (Root Mean Squared Logarithmic Error): Handles the skewed distribution of trip durations

### Model Choices

**1. Random Forest Regressor**
- **Pros**: Robust to outliers, handles non-linear relationships, requires minimal hyperparameter tuning
- **Cons**: Larger model size, slower inference
- **Configuration**: 50 trees, max depth 8, min samples split 5

**2. XGBoost Regressor** (Best Model)
- **Pros**: Superior performance, handles missing values well, regularization prevents overfitting
- **Cons**: Requires more careful tuning
- **Configuration**: 50 estimators, max depth 6, learning rate 0.1
- **Performance**: 
  - Validation RMSE: 341.49 seconds (~5.7 minutes)
  - Validation R²: 0.6996
  - Validation MAE: 193.12 seconds (~3.2 minutes)

**Why XGBoost Won**: XGBoost achieved ~9% lower RMSE and ~11% lower MAE compared to Random Forest, making it the production model.

### Feature Engineering

**Time Features:**
- Hour of day, day of week, month, year
- Is weekend, is rush hour (7-9 AM, 5-7 PM)
- Is night time (10 PM - 5 AM)

**Distance Features:**
- Haversine distance (great circle distance)
- Manhattan distance (grid-based)
- Bearing/direction of travel

**Categorical Encodings:**
- Vendor ID
- Passenger count

## Getting Started

### Prerequisites

- **Python**: 3.11+
- **uv**: Fast Python package installer (recommended) or pip
- **Docker**: For containerized execution
- **AWS Account**: For SageMaker deployment (optional)

### Installation

#### Option 1: Using uv (Recommended)

```bash
# Install uv if not already installed
pip install uv

# Clone the repository
git clone https://github.com/CMMouaikel/mlops2025_christa_nicolas.git
cd mlops2025_christa_nicolas

# Install dependencies
uv pip install -e .
```

#### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/CMMouaikel/mlops2025_christa_nicolas.git
cd mlops2025_christa_nicolas

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

##  Running Locally

### 1. Complete Training Pipeline

Run the entire pipeline (preprocessing → feature engineering → training):

```bash
uv run train
```

### 2. Step-by-Step Execution

#### Step 1: Data Preprocessing

```bash
uv run python scripts/preprocess.py \
  --config configs/train.yaml \
  --input src/mlproject/data/train.csv \
  --output src/mlproject/data/processed/train_cleaned.csv
```

**Output**: Cleaned data in `src/mlproject/data/processed/`

#### Step 2: Feature Engineering

```bash
uv run python scripts/feature_engineering.py \
  --config configs/train.yaml \
  --input src/mlproject/data/processed/train_cleaned.csv \
  --output src/mlproject/data/features/train_features.csv
```

**Output**: 
- Features in `src/mlproject/data/features/train_features.csv`
- Fitted encoder in `src/mlproject/data/features/feature_engineer.pkl`

#### Step 3: Model Training

```bash
uv run python scripts/train.py \
  --config configs/train.yaml \
  --input src/mlproject/data/features/train_features.csv
```

**Output**:
- Models in `outputs/models/` (best_model.pkl, random_forest.pkl, xgboost.pkl)
- Metrics in `outputs/metrics/training_metrics.json`

### 3. Batch Inference

Run the complete inference pipeline:

```bash
uv run inference --input src/mlproject/data/test.csv
```
Or run with specific options:

```bash
uv run inference \
  --input src/mlproject/data/test.csv \
  --model outputs/models/best_model.pkl \
  --feature-engineer src/mlproject/data/features/feature_engineer.pkl \
  --output output/predictions.csv
```

**Output**: Predictions in `output/predictions.csv`

### 4. Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_preprocess.py -v

# Run with coverage
uv run pytest tests/ --cov=src/mlproject --cov-report=html
```

## Running with Docker

### Build the Docker Image

```bash
docker build -t mlops-nyc-taxi .
```

### Run with Docker Compose

#### 1. Start the Container

```bash
docker-compose up -d
```

#### 2. Execute Training Pipeline

```bash
# Run complete pipeline
docker-compose exec app uv run train
# Or run individual steps
docker-compose exec app python scripts/preprocess.py
docker-compose exec app python scripts/feature_engineering.py --input src/mlproject/data/processed/train_cleaned.csv
docker-compose exec app python scripts/train.py --input src/mlproject/data/features/train_features.csv
```

#### 3. Run Batch Inference

```bash
#Run complete inference pipeline
docker-compose exec app python uv run inference --input ssrc/mlproject/data/test_sample.csv \

#Or with specific options
docker-compose exec app uv run inference \
  --input src/mlproject/data/test_sample.csv \
  --model outputs/models/best_model.pkl \
  --feature-engineer src/mlproject/data/features/feature_engineer.pkl
```

#### 4. Access Interactive Shell

```bash
docker-compose exec app /bin/bash
```

#### 5. Stop the Container

```bash
docker-compose down
```

### Volume Mounts

The Docker setup mounts the following directories for persistence:
- `./outputs:/app/outputs` - Trained models and metrics
- `./src/mlproject/data:/app/src/mlproject/data` - Data files
- `./configs:/app/configs` - Configuration files
- `./scripts:/app/scripts` - Scripts (for development)

## Running on AWS SageMaker

### Prerequisites

1. **AWS Account** with SageMaker access
2. **S3 Bucket** for data and model artifacts
3. **IAM Role** with SageMaker permissions
4. **AWS CLI** configured with credentials

### Setup

1. **Configure SageMaker Settings**

Edit `configs/sagemaker.yaml`:

```yaml
sagemaker:
  role: "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole"
  bucket: "your-s3-bucket-name"
  region: "us-east-1"
  instance_type: "ml.m5.xlarge"
  instance_count: 1
```

2. **Upload Data to S3**

```bash
# Upload training data
aws s3 cp src/mlproject/data/train.csv s3://your-bucket/mlops-nyc-taxi/data/train.csv

# Upload test data
aws s3 cp src/mlproject/data/test.csv s3://your-bucket/mlops-nyc-taxi/data/test.csv
```

### Running Training Pipeline on SageMaker

```bash
uv run train \
  --env sagemaker \
  --config configs/train.yaml \
  --sagemaker-config configs/sagemaker.yaml \
  --input s3://your-bucket/mlops-nyc-taxi/data/train.csv
```

**What Happens:**
1. Pipeline creates SageMaker Processing Jobs for preprocessing and feature engineering
2. Creates SageMaker Training Job for model training
3. Stores artifacts in S3 at `s3://your-bucket/mlops-nyc-taxi/outputs/`
4. Registers the best model in SageMaker Model Registry

### Running Batch Inference on SageMaker

```bash
uv run inference \
  --env sagemaker \
  --sagemaker-config configs/sagemaker.yaml \
  --input s3://your-bucket/mlops-nyc-taxi/data/test.csv \
  --model-name mlops-nyc-taxi-xgboost \
  --output s3://your-bucket/mlops-nyc-taxi/predictions/
```

### Monitoring SageMaker Jobs

```bash
# View training jobs
aws sagemaker list-training-jobs --max-results 10

# View processing jobs
aws sagemaker list-processing-jobs --max-results 10

# View job logs
aws logs tail /aws/sagemaker/TrainingJobs --follow
```

### SageMaker Pipeline Architecture

```
S3 Input Data
    ↓
SageMaker Processing Job (Preprocessing)
    ↓
SageMaker Processing Job (Feature Engineering)
    ↓
SageMaker Training Job (Model Training)
    ↓
Model Registry & S3 Artifacts
    ↓
SageMaker Batch Transform (Inference)
    ↓
S3 Predictions Output
```

## Performance Results

### Training Set (10,000 samples)
- **Random Forest**: RMSE 3069.49s, R² 0.078
- **XGBoost**: RMSE 3070.04s, R² 0.078

### Validation Set (2,000 samples)
- **Random Forest**: RMSE 377.08s, MAE 216.22s, R² 0.634
- **XGBoost**: RMSE 341.49s, MAE 193.12s, R² 0.700 

### Inference Performance
- **Mean Prediction**: 13.1 minutes
- **Prediction Range**: 2.1 - 52.6 minutes
- **Throughput**: ~1000 predictions in <1 second

## Configuration

All pipeline behavior is controlled via `configs/train.yaml`:

```yaml
data:
  train_path: "src/mlproject/data/train.csv"
  test_path: "src/mlproject/data/test.csv"
  processed_dir: "src/mlproject/data/processed"
  features_dir: "src/mlproject/data/features"

preprocessing:
  drop_zero_duration: true
  max_duration_hours: 24
  speed_threshold_mph: 100

model:
  output_dir: "outputs/models"
  metrics_dir: "outputs/metrics"
  models:
    - name: "random_forest"
      params:
        n_estimators: 50
        max_depth: 8
    - name: "xgboost"
      params:
        n_estimators: 50
        max_depth: 6
        learning_rate: 0.1
```


## 

##  Team Contributions

### Nicolas Azar
- **Repository setup + branching workflow**
  - Pulled the repository and managed development work across branches to keep changes isolated and clean.
  - Checked out the `feat/features` branch and ensured the project structure was ready for the new module.
- **Feature engineering module implementation**
  - Created the `src/mlproject/features/` package and added the complete `features` folder inside `src/mlproject/`.
  - Added/updated the `features:` configuration block in `configs/train.yaml`, defining:
    - **Time features**: `hour`, `day_of_week`, `month`, `is_weekend`
    - **Distance features**: `haversine_distance`, `manhattan_distance`
    - **Categorical features**: `vendor_id`, `passenger_count`
- **Feature engineering execution script**
  - Implemented the standalone script `scripts/feature_engineering.py` to run feature engineering independently (CLI-friendly).
  - Ensured the script works with the YAML config and integrates properly with the data flow (input cleaned data → output features).
- **Testing**
  - Added a dedicated test file `tests/test_features.py` to validate feature engineering behavior and catch regressions early.
- **Version control discipline**
  - Added all new files, committed, and pushed work to the remote branch using `git push origin <branch_name>`.
  
- **Training configuration + training branch work**
  - Switched back to `master` and created a new branch `feat/train`.
  - Extended `configs/train.yaml` with the **training/model configuration**, including:
    - Output directories for models and metrics
    - Two models: **Random Forest** and **XGBoost**, with explicit hyperparameters
    - Training parameters: `test_size`, `random_state`, and selected metric (`rmse`)
  - Created the `src/mlproject/train/` module and added a training entry script `scripts/train.py`.
- **Dependencies + documentation**
  - Added the `xgboost` dependency (`uv add xgboost`) to support training the boosting model.
  - Contributed to README updates and ensured documentation matched the new modules and scripts.

---

### Christa Maria Mouaikel
- **End-to-end pipeline completion**
  - Completed the remaining project components and ensured the pipeline runs as a full workflow from raw data → processed data → engineered features → trained model → saved metrics/artifacts.
- **Preprocessing + data validation**
  - Implemented and finalized the preprocessing/cleaning logic and ensured it integrates with the rest of the pipeline consistently.
  - Helped ensure the pipeline handles realistic dataset issues (missing values, outliers, invalid durations, etc.) and produces stable outputs.
- **Inference + batch prediction**
  - Implemented or finalized the batch inference workflow so predictions can be generated reproducibly from a saved model + feature pipeline.
  - Ensured outputs are saved in a clear format and align with project expectations.
- **Dockerization**
  - Built and refined the Docker setup (Dockerfile / docker-compose) so the project can run in a container with correct dependencies and volume mounts.
  - Ensured “Docker run” instructions match the actual container behavior.
- **SageMaker integration**
  - Completed the AWS/SageMaker pipeline execution path and configuration, enabling training/inference in the cloud environment.
  - Ensured configs and scripts support SageMaker runs in addition to local and Docker runs.
- **Testing + documentation polish**
  - Completed remaining tests and improved documentation for clarity and reproducibility (run commands, expected artifacts, outputs folders, etc.).
  - Ensured the repository reads well for evaluators: clear setup, repeatable runs, and clean project structure.

*Note: While each member owned specific components, final integration required collaboration to ensure the whole pipeline executed correctly end-to-end. All parts where planned ahead and distributed as such.*



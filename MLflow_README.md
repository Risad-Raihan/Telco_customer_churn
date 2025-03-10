# MLflow Integration for Telco Customer Churn Prediction

This guide explains how to use MLflow with the Telco Customer Churn Prediction model.

## What is MLflow?

MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It includes:

- **Tracking**: Record and query experiments (code, data, config, results)
- **Projects**: Package data science code in a reusable, reproducible form
- **Models**: Deploy ML models in diverse serving environments
- **Model Registry**: Store, annotate, discover, and manage models in a central repository

## Why Use MLflow?

MLflow helps you:

1. **Track experiments**: Compare different models and parameters
2. **Reproduce results**: Ensure consistency across runs
3. **Share models**: Make it easy for others to use your models
4. **Deploy models**: Streamline the path to production
5. **Version control**: Keep track of model versions

## Installation

Install MLflow using pip:

```bash
pip install mlflow
```

## Getting Started

The `churn_prediction_mlflow.py` script demonstrates how to integrate MLflow with your churn prediction model. Here's what it does:

1. Sets up MLflow tracking
2. Loads and preprocesses the data
3. Trains multiple models (Decision Tree, Random Forest, XGBoost)
4. Logs parameters, metrics, and artifacts for each model
5. Saves the best model

## Running the Script

```bash
python churn_prediction_mlflow.py
```

## Viewing the Results

After running the script, you can view the results using the MLflow UI:

```bash
mlflow ui
```

Then open your browser and go to http://localhost:5000

## Key MLflow Concepts Used

### 1. Tracking Experiments

```python
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Telco Customer Churn Prediction")
```

### 2. Logging Parameters

```python
for param_name, param_value in model.get_params().items():
    mlflow.log_param(param_name, param_value)
```

### 3. Logging Metrics

```python
mlflow.log_metric("test_accuracy", test_accuracy)
mlflow.log_metric("roc_auc", roc_auc)
```

### 4. Logging Artifacts (plots, files)

```python
mlflow.log_artifact(cm_plot_path)
```

### 5. Logging Models

```python
signature = infer_signature(X_test, y_test_pred)
mlflow.sklearn.log_model(model, model_name, signature=signature)
```

## Loading Models from MLflow

You can load models saved with MLflow:

```python
# Replace <run_id> with the actual run ID from the MLflow UI
model_uri = 'runs:/<run_id>/random_forest_model'
loaded_model = mlflow.sklearn.load_model(model_uri)
```

## Integrating with the Streamlit App

To use MLflow models in your Streamlit app:

1. Load the model from MLflow instead of from a pickle file:

```python
import mlflow.sklearn

# Replace with your actual run ID
run_id = "your_run_id"
model_uri = f"runs:/{run_id}/random_forest_model"
model = mlflow.sklearn.load_model(model_uri)
```

2. Use the loaded model for predictions:

```python
prediction = model.predict(input_data_df)
pred_prob = model.predict_proba(input_data_df)
```

## Advanced MLflow Features

### Model Registry

For production deployments, use the MLflow Model Registry:

```python
# Register the model
model_name = "TelcoChurnModel"
mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name)

# Load a specific version
model_version = 1
model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
```

### Hyperparameter Tuning

Combine MLflow with hyperparameter tuning:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}

with mlflow.start_run():
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Log the best parameters and model
    mlflow.log_params(grid_search.best_params_)
    mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model")
```

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow GitHub Repository](https://github.com/mlflow/mlflow)
- [MLflow Tutorials](https://mlflow.org/docs/latest/tutorials-and-examples/index.html)

## Running the MLflow UI

```bash
run_mlflow_ui.bat
```

Then open your browser and go to http://localhost:5000 
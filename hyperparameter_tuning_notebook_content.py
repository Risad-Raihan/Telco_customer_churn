# %% [markdown]
# # Telco Customer Churn Prediction with Hyperparameter Tuning and MLflow
# 
# This notebook demonstrates how to perform hyperparameter tuning for machine learning models and track experiments with MLflow.

# %% [markdown]
# ## 1. Import Libraries
# 
# First, let's import all the necessary libraries.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, auc, f1_score
import pickle
import os
import time
from datetime import datetime

# For Google Colab, you might need to install MLflow first
# !pip install mlflow

# Import MLflow libraries
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# %% [markdown]
# ## 2. Set Up MLflow
# 
# Configure MLflow to track our experiments. For Google Colab, we'll use a local directory for tracking.

# %%
# Set MLflow tracking URI - this can be a local directory or a remote server
# For local tracking, we'll use a directory called 'mlruns'
mlflow.set_tracking_uri("file:./mlruns")

# Set the experiment name
mlflow.set_experiment("Telco Customer Churn Hyperparameter Tuning")

print("="*80)
print("HYPERPARAMETER TUNING FOR CHURN PREDICTION MODEL")
print("="*80)

# %% [markdown]
# ## 3. Load and Preprocess Data
# 
# Now, let's load the dataset and perform necessary preprocessing steps.

# %%
print("\n1. LOADING AND PREPROCESSING DATA")
print("-"*50)

# For Google Colab, you might need to upload the dataset or download it
# Uncomment the following lines if you're using Google Colab and need to download the dataset
# !wget -O WA_Fn-UseC_-Telco-Customer-Churn.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

# Load the CSV data to a pandas dataframe
try:
    # Try to load from the Data directory first
    data_path = "Data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(data_path)
except FileNotFoundError:
    # If not found, try the current directory
    data_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(data_path)

print(f"Dataset loaded with shape: {df.shape}")

# Display the first few rows
df.head()

# %% 
# Data preprocessing
# Drop customer ID column
df = df.drop(columns=["customerID"])
print("Dropped customerID column")

# Handle missing values in TotalCharges
df["TotalCharges"] = df['TotalCharges'].replace({" ":"0.0"})
df["TotalCharges"] = df['TotalCharges'].astype(float)
print("Handled missing values in TotalCharges")

# Label encoding for target column
df["Churn"] = df["Churn"].replace({"Yes":1, "No":0})
print("Encoded target column 'Churn'")

# Check class distribution
print("\nClass distribution:")
print(df["Churn"].value_counts())
print(f"Churn rate: {df['Churn'].mean():.2%}")

# %% 
# Label encoding for categorical features
object_columns = df.select_dtypes(include="object").columns
encoders = {}

# Apply label encoding and store the encoders
for column in object_columns:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    encoders[column] = label_encoder

# Save the encoders to a pickle file
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print(f"\nEncoded {len(object_columns)} categorical features")
print("Encoders saved to 'encoders.pkl'")

# %%
# Split the features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Apply SMOTE for handling class imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"SMOTE-resampled training set shape: {X_train_smote.shape}")
print(f"Class distribution after SMOTE: {np.bincount(y_train_smote)}")

# %% [markdown]
# ## 4. Understanding Hyperparameter Tuning
# 
# Before we start tuning, let's understand what hyperparameter tuning is and why it's important.

# %%
print("\n2. WHAT IS HYPERPARAMETER TUNING?")
print("-"*50)
print("""
Hyperparameter tuning is the process of finding the optimal set of hyperparameters for a machine learning model.
Hyperparameters are parameters that are not learned from the data but are set before training.

Why is hyperparameter tuning important?
- Improves model performance
- Prevents overfitting or underfitting
- Helps find the best model configuration for your specific problem

In this notebook, we'll use two methods for hyperparameter tuning:
1. GridSearchCV: Exhaustively searches through a specified parameter grid
2. RandomizedSearchCV: Samples a specified number of combinations from the parameter space

We'll track all experiments with MLflow to compare different parameter combinations.
""")

# %% [markdown]
# ## 5. Define Hyperparameter Search Spaces
# 
# For each model, we need to define the hyperparameters we want to tune and the values to try.

# %%
print("\n3. DEFINING HYPERPARAMETER SEARCH SPACES")
print("-"*50)

# Define hyperparameter search spaces for each model
print("Defining hyperparameter search spaces for each model...")

# Decision Tree hyperparameters
dt_param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}
print("\nDecision Tree hyperparameters:")
for param, values in dt_param_grid.items():
    print(f"- {param}: {values}")
print(f"Total combinations: {np.prod([len(v) for v in dt_param_grid.values()])}")

# Random Forest hyperparameters
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
print("\nRandom Forest hyperparameters:")
for param, values in rf_param_grid.items():
    print(f"- {param}: {values}")
print(f"Total combinations: {np.prod([len(v) for v in rf_param_grid.values()])}")

# XGBoost hyperparameters
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}
print("\nXGBoost hyperparameters:")
for param, values in xgb_param_grid.items():
    print(f"- {param}: {values}")
print(f"Total combinations: {np.prod([len(v) for v in xgb_param_grid.values()])}")

# %% [markdown]
# ## 6. Create Hyperparameter Tuning Function
# 
# Now, let's create a function that will perform hyperparameter tuning and log the results to MLflow.

# %%
print("\n4. HYPERPARAMETER TUNING WITH MLFLOW TRACKING")
print("-"*50)

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Function to perform hyperparameter tuning and log results to MLflow
def tune_and_log(model_name, model, param_grid, X_train, y_train, X_test, y_test, search_method='grid', n_iter=10):
    print(f"\nTuning hyperparameters for {model_name} using {search_method.capitalize()}SearchCV...")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{model_name} - {search_method.capitalize()} Search"):
        # Log basic information
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("search_method", search_method)
        mlflow.log_param("cv_folds", cv.n_splits)
        
        # Create the search object
        if search_method == 'grid':
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring='f1',  # Using F1 score as it balances precision and recall
                n_jobs=-1,     # Use all available cores
                verbose=1
            )
            mlflow.log_param("total_combinations", np.prod([len(v) for v in param_grid.values()]))
        else:  # randomized search
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
            mlflow.log_param("n_iter", n_iter)
        
        # Record start time
        start_time = time.time()
        
        # Fit the search object to the data
        search.fit(X_train, y_train)
        
        # Record end time and calculate duration
        end_time = time.time()
        duration = end_time - start_time
        mlflow.log_metric("search_duration_seconds", duration)
        print(f"Search completed in {duration:.2f} seconds")
        
        # Get the best parameters and score
        best_params = search.best_params_
        best_score = search.best_score_
        
        print(f"\nBest parameters found:")
        for param, value in best_params.items():
            print(f"- {param}: {value}")
            mlflow.log_param(f"best_{param}", value)
        
        print(f"Best cross-validation F1 score: {best_score:.4f}")
        mlflow.log_metric("best_cv_f1_score", best_score)
        
        # Get the best model
        best_model = search.best_estimator_
        
        # Evaluate on test set
        y_test_pred = best_model.predict(X_test)
        y_test_prob = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_roc_auc = roc_auc_score(y_test, y_test_prob)
        
        # Calculate Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
        test_pr_auc = auc(recall, precision)
        
        # Log metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        mlflow.log_metric("test_pr_auc", test_pr_auc)
        
        # Generate classification report
        report = classification_report(y_test, y_test_pred, output_dict=True)
        mlflow.log_metric("test_precision", report['1']['precision'])
        mlflow.log_metric("test_recall", report['1']['recall'])
        
        # Print test results
        print(f"\nTest set results with best parameters:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"F1 Score: {test_f1:.4f}")
        print(f"ROC AUC: {test_roc_auc:.4f}")
        print(f"PR AUC: {test_pr_auc:.4f}")
        print(f"Classification Report:\n{classification_report(y_test, y_test_pred)}")
        
        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name} (Best Parameters)')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Save confusion matrix plot
        cm_plot_path = f"confusion_matrix_{model_name}_tuned.png"
        plt.savefig(cm_plot_path)
        plt.close()
        
        # Log the confusion matrix plot
        mlflow.log_artifact(cm_plot_path)
        
        # Create feature importance plot (if available)
        if hasattr(best_model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            feature_importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            sns.barplot(x='Importance', y='Feature', data=feature_importance)
            plt.title(f'Feature Importance - {model_name} (Best Parameters)')
            
            # Save feature importance plot
            fi_plot_path = f"feature_importance_{model_name}_tuned.png"
            plt.savefig(fi_plot_path)
            plt.close()
            
            # Log the feature importance plot
            mlflow.log_artifact(fi_plot_path)
        
        # Log the best model
        signature = infer_signature(X_test, y_test_pred)
        mlflow.sklearn.log_model(best_model, f"{model_name}_best_model", signature=signature)
        
        # Log top CV results as artifacts
        cv_results_df = pd.DataFrame(search.cv_results_)
        cv_results_df = cv_results_df.sort_values('rank_test_score').head(10)
        cv_results_path = f"{model_name}_top_cv_results.csv"
        cv_results_df.to_csv(cv_results_path, index=False)
        mlflow.log_artifact(cv_results_path)
        
        # Clean up temporary files
        if os.path.exists(cm_plot_path):
            os.remove(cm_plot_path)
        if hasattr(best_model, 'feature_importances_') and os.path.exists(fi_plot_path):
            os.remove(fi_plot_path)
        if os.path.exists(cv_results_path):
            os.remove(cv_results_path)
        
        return best_model, best_params, test_f1

# %% [markdown]
# ## 7. Perform Hyperparameter Tuning for Each Model
# 
# Now, let's tune the hyperparameters for each model and track the results with MLflow.

# %%
# Perform hyperparameter tuning for each model
print("\nStarting hyperparameter tuning for each model...")

# Decision Tree - Grid Search
dt_model = DecisionTreeClassifier(random_state=42)
dt_best_model, dt_best_params, dt_f1 = tune_and_log(
    "Decision Tree", 
    dt_model, 
    dt_param_grid, 
    X_train_smote, 
    y_train_smote, 
    X_test, 
    y_test, 
    search_method='grid'
)

# %% 
# Random Forest - Randomized Search (due to large search space)
rf_model = RandomForestClassifier(random_state=42)
rf_best_model, rf_best_params, rf_f1 = tune_and_log(
    "Random Forest", 
    rf_model, 
    rf_param_grid, 
    X_train_smote, 
    y_train_smote, 
    X_test, 
    y_test, 
    search_method='randomized',
    n_iter=20
)

# %%
# XGBoost - Randomized Search (due to large search space)
xgb_model = XGBClassifier(random_state=42)
xgb_best_model, xgb_best_params, xgb_f1 = tune_and_log(
    "XGBoost", 
    xgb_model, 
    xgb_param_grid, 
    X_train_smote, 
    y_train_smote, 
    X_test, 
    y_test, 
    search_method='randomized',
    n_iter=20
)

# %% [markdown]
# ## 8. Compare Models and Select the Best
# 
# Now, let's compare the performance of all models and select the best one.

# %%
print("\n5. COMPARING MODELS AND SELECTING THE BEST")
print("-"*50)

# Compare models based on F1 score
models_comparison = {
    "Decision Tree": {"model": dt_best_model, "params": dt_best_params, "f1": dt_f1},
    "Random Forest": {"model": rf_best_model, "params": rf_best_params, "f1": rf_f1},
    "XGBoost": {"model": xgb_best_model, "params": xgb_best_params, "f1": xgb_f1}
}

# Find the best model
best_model_name = max(models_comparison, key=lambda x: models_comparison[x]["f1"])
best_model_info = models_comparison[best_model_name]

print(f"\nModel comparison based on test F1 score:")
for model_name, info in models_comparison.items():
    print(f"- {model_name}: {info['f1']:.4f}")

print(f"\nBest model: {best_model_name} with F1 score of {best_model_info['f1']:.4f}")
print(f"Best parameters: {best_model_info['params']}")

# %% [markdown]
# ## 9. Save the Best Model
# 
# Let's save the best model with MLflow and as a pickle file.

# %%
# Save the best model with MLflow
with mlflow.start_run(run_name=f"Best Tuned Model - {best_model_name}"):
    # Log model parameters
    for param_name, param_value in best_model_info["params"].items():
        mlflow.log_param(param_name, param_value)
    
    # Make predictions
    y_test_pred = best_model_info["model"].predict(X_test)
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    # Log metrics
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1_score", test_f1)
    
    # Log the model
    model_data = {
        "model": best_model_info["model"], 
        "features_names": X.columns.tolist(),
        "best_params": best_model_info["params"]
    }
    
    # Save the model to a pickle file
    model_filename = f"customer_churn_model_tuned_{best_model_name.lower().replace(' ', '_')}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model_data, f)
    
    # Log the pickle file as an artifact
    mlflow.log_artifact(model_filename)
    
    # Log the model with MLflow
    signature = infer_signature(X_test, y_test_pred)
    mlflow.sklearn.log_model(best_model_info["model"], "best_tuned_model", signature=signature)
    
    print(f"\nBest tuned model ({best_model_name}) saved successfully!")
    print(f"Model saved to: {model_filename}")
    print(f"Model accuracy: {test_accuracy:.4f}")
    print(f"Model F1 score: {test_f1:.4f}")

# %% [markdown]
# ## 10. What We've Learned from Hyperparameter Tuning
# 
# Let's summarize what we've learned from the hyperparameter tuning process.

# %%
print("\n6. WHAT WE'VE LEARNED FROM HYPERPARAMETER TUNING")
print("-"*50)
print("""
Key insights from hyperparameter tuning:

1. Impact of hyperparameters:
   - We've seen how different hyperparameter values affect model performance
   - Some parameters have more influence than others

2. Model comparison:
   - We can now make an informed decision about which model works best for our problem
   - We understand the trade-offs between different models

3. Preventing overfitting:
   - Proper hyperparameter tuning helps prevent overfitting
   - Cross-validation ensures our model generalizes well to unseen data

4. MLflow tracking:
   - We've tracked all experiments with MLflow
   - We can easily compare different runs and parameter combinations
   - All models, metrics, and artifacts are stored for future reference
""")

print("\nMLflow tracking URI:", mlflow.get_tracking_uri())
print("Run 'mlflow ui' in the terminal to view the experiment results in a web UI")

# %% [markdown]
# ## 11. How to Use the Tuned Model
# 
# Finally, let's see how to use the tuned model for making predictions.

# %%
print("\n7. HOW TO USE THE TUNED MODEL")
print("-"*50)
print(f"""
To use the tuned model for predictions:

1. Load the model:
   ```python
   import pickle
   with open("{model_filename}", "rb") as f:
       model_data = pickle.load(f)
   
   tuned_model = model_data["model"]
   ```

2. Or load directly from MLflow:
   ```python
   import mlflow.sklearn
   model_uri = "runs:/<run_id>/best_tuned_model"
   tuned_model = mlflow.sklearn.load_model(model_uri)
   ```

3. Prepare input data (ensure it's encoded the same way as during training)

4. Make predictions:
   ```python
   predictions = tuned_model.predict(input_data)
   probabilities = tuned_model.predict_proba(input_data)[:, 1]
   ```
""")

# %% [markdown]
# ## 12. For Google Colab: Access MLflow UI
# 
# If you're running this notebook in Google Colab, you can use ngrok to access the MLflow UI.

# %%
# Install and set up ngrok for accessing MLflow UI from Colab
# Uncomment these lines if you're using Google Colab

# !pip install pyngrok
# from pyngrok import ngrok
# !mlflow ui --port 5000 &
# public_url = ngrok.connect(5000)
# print(f"MLflow UI available at: {public_url}")

# %% [markdown]
# ## 13. Example: Making Predictions with the Tuned Model
# 
# Let's demonstrate how to use the tuned model to make predictions on new data.

# %%
# Example input data
input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

# Convert to DataFrame
input_data_df = pd.DataFrame([input_data])

# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Encode categorical features
for column, encoder in encoders.items():
    if column in input_data_df.columns:
        input_data_df[column] = encoder.transform(input_data_df[column])

# Make prediction using the best model
best_model = best_model_info["model"]
prediction = best_model.predict(input_data_df)
pred_prob = best_model.predict_proba(input_data_df)

# Display results
print(f"\nPrediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
print(f"Probability of churning: {pred_prob[0][1]:.2%}")

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we've demonstrated how to:
# 
# 1. Set up MLflow for experiment tracking
# 2. Define hyperparameter search spaces for different models
# 3. Perform hyperparameter tuning using Grid Search and Randomized Search
# 4. Track all experiments, parameters, metrics, and artifacts with MLflow
# 5. Compare models and select the best one
# 6. Save and use the tuned model for making predictions
# 
# Hyperparameter tuning is a crucial step in the machine learning workflow that can significantly improve model performance. MLflow makes it easy to track and compare different experiments, helping you make informed decisions about which model and parameters to use. 
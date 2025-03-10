# %% [markdown]
# # Telco Customer Churn Prediction with MLflow
# 
# This notebook demonstrates how to integrate MLflow with a machine learning model for customer churn prediction. MLflow helps track experiments, manage models, and simplify the deployment process.
# 
# ## Setup
# 
# First, let's install the required packages. If you're running this on Google Colab, you'll need to install MLflow and other dependencies.

# %%
# Install required packages
# !pip install mlflow scikit-learn pandas numpy matplotlib seaborn xgboost imbalanced-learn

# %% [markdown]
# ## Import Libraries
# 
# Now, let's import all the necessary libraries for our project.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, auc
import pickle
import os

# Import MLflow libraries
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# %% [markdown]
# ## Set Up MLflow
# 
# Configure MLflow to track our experiments. For Google Colab, we'll use a local directory for tracking.

# %%
# Set MLflow tracking URI - this can be a local directory or a remote server
# For local tracking, we'll use a directory called 'mlruns'
mlflow.set_tracking_uri("file:./mlruns")

# Set the experiment name
mlflow.set_experiment("Telco Customer Churn Prediction")

# Print the tracking URI to confirm
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

# %% [markdown]
# ## Data Loading and Preprocessing
# 
# Let's load the dataset and perform necessary preprocessing steps.

# %%
# For Google Colab, you might need to upload the dataset or download it
# Uncomment the following lines if you're using Google Colab and need to download the dataset

# !wget -O WA_Fn-UseC_-Telco-Customer-Churn.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

# Load the CSV data to a pandas dataframe
# Adjust the path as needed for your environment
try:
    # Try to load from the Data directory first
    data_path = "Data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(data_path)
except FileNotFoundError:
    # If not found, try the current directory
    data_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(data_path)

# Display the first few rows of the dataset
df.head()

# %%
# Check the shape of the dataset
print(f"Dataset shape: {df.shape}")

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# %%
# Data preprocessing
# Drop customer ID column
df = df.drop(columns=["customerID"])

# Handle missing values in TotalCharges
df["TotalCharges"] = df['TotalCharges'].replace({" ":"0.0"})
df["TotalCharges"] = df['TotalCharges'].astype(float)

# Label encoding for target column
df["Churn"] = df["Churn"].replace({"Yes":1, "No":0})

# Check class distribution
print("\nClass distribution:")
print(df["Churn"].value_counts())
print(f"Churn rate: {df['Churn'].mean():.2%}")

# %% [markdown]
# ## Exploratory Data Analysis (EDA)
# 
# Let's visualize some key aspects of the data to better understand it.

# %%
# Function to plot histograms for numerical features
def plot_histogram(df, column_name):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column_name], kde=True)
    plt.title(f"Distribution of {column_name}")
    
    # Calculate mean and median
    col_mean = df[column_name].mean()
    col_median = df[column_name].median()
    
    # Add vertical lines for mean and median
    plt.axvline(col_mean, color="red", linestyle="--", label="Mean")
    plt.axvline(col_median, color="green", linestyle="-", label="Median")
    
    plt.legend()
    plt.show()

# Plot histograms for numerical features
numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
for feature in numerical_features:
    plot_histogram(df, feature)

# %%
# Correlation heatmap for numerical features
plt.figure(figsize=(10, 8))
sns.heatmap(df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]].corr(), 
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# %%
# Churn rate by contract type
plt.figure(figsize=(10, 6))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn by Contract Type")
plt.show()

# %% [markdown]
# ## Feature Engineering and Preprocessing
# 
# Now, let's encode categorical features and prepare the data for modeling.

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

print(f"Encoded {len(object_columns)} categorical features")
print(f"Encoders saved to 'encoders.pkl'")

# %%
# Split the features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Training set class distribution:\n{y_train.value_counts()}")

# %%
# Apply SMOTE for handling class imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"SMOTE-resampled training set shape: {X_train_smote.shape}")
print(f"SMOTE-resampled training set class distribution:\n{y_train_smote.value_counts()}")

# %% [markdown]
# ## Model Training with MLflow Tracking
# 
# Now, let's train multiple models and track them with MLflow.

# %%
# Define models to train
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

# %%
# Train and log each model with MLflow
for model_name, model in models.items():
    # Start an MLflow run
    with mlflow.start_run(run_name=model_name):
        print(f"\n{'='*50}\nTraining {model_name}\n{'='*50}")
        
        # Log model parameters
        for param_name, param_value in model.get_params().items():
            mlflow.log_param(param_name, param_value)
        
        # Train the model
        model.fit(X_train_smote, y_train_smote)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Calculate ROC AUC
        roc_auc = roc_auc_score(y_test, y_test_prob)
        
        # Calculate Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
        pr_auc = auc(recall, precision)
        
        # Generate classification report
        report = classification_report(y_test, y_test_pred, output_dict=True)
        
        # Log metrics to MLflow
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("precision", report['1']['precision'])
        mlflow.log_metric("recall", report['1']['recall'])
        mlflow.log_metric("f1_score", report['1']['f1-score'])
        
        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Save confusion matrix plot
        cm_plot_path = f"confusion_matrix_{model_name}.png"
        plt.savefig(cm_plot_path)
        plt.show()
        
        # Log the confusion matrix plot
        mlflow.log_artifact(cm_plot_path)
        
        # Create feature importance plot (if available)
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            sns.barplot(x='Importance', y='Feature', data=feature_importance)
            plt.title(f'Feature Importance - {model_name}')
            
            # Save feature importance plot
            fi_plot_path = f"feature_importance_{model_name}.png"
            plt.savefig(fi_plot_path)
            plt.show()
            
            # Log the feature importance plot
            mlflow.log_artifact(fi_plot_path)
        
        # Log the model with its signature
        signature = infer_signature(X_test, y_test_pred)
        mlflow.sklearn.log_model(model, model_name, signature=signature)
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC: {pr_auc:.4f}")
        print(f"Classification Report:\n{classification_report(y_test, y_test_pred)}")
        
        # Clean up temporary files
        if os.path.exists(cm_plot_path):
            os.remove(cm_plot_path)
        if hasattr(model, 'feature_importances_') and os.path.exists(fi_plot_path):
            os.remove(fi_plot_path)

# %% [markdown]
# ## Save the Best Model
# 
# Based on the results, let's save the best model (Random Forest) with MLflow.

# %%
# Select the best model (Random Forest in this case)
best_model = models["Random Forest"]

# Save the best model with MLflow
with mlflow.start_run(run_name="Best Model - Random Forest"):
    # Log model parameters
    for param_name, param_value in best_model.get_params().items():
        mlflow.log_param(param_name, param_value)
    
    # Make predictions
    y_test_pred = best_model.predict(X_test)
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Log metrics
    mlflow.log_metric("test_accuracy", test_accuracy)
    
    # Log the model
    model_data = {"model": best_model, "features_names": X.columns.tolist()}
    
    # Save the model to a pickle file
    with open("customer_churn_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    # Log the pickle file as an artifact
    mlflow.log_artifact("customer_churn_model.pkl")
    
    # Log the model with MLflow
    signature = infer_signature(X_test, y_test_pred)
    mlflow.sklearn.log_model(best_model, "random_forest_model", signature=signature)
    
    print("\nBest Model (Random Forest) saved successfully!")
    print(f"Model accuracy: {test_accuracy:.4f}")

# %% [markdown]
# ## View MLflow Tracking Information
# 
# Let's check the MLflow tracking information and see how to access the UI.

# %%
print("\nMLflow tracking URI:", mlflow.get_tracking_uri())
print("Run 'mlflow ui' in the terminal to view the experiment results in a web UI")

# %% [markdown]
# ## For Google Colab: Access MLflow UI
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
# ## Loading a Model from MLflow
# 
# Here's how to load a model from MLflow for making predictions.

# %%
# Example of how to load a model from MLflow
print("\nExample of how to load a model from MLflow:")
print("model_uri = 'runs:/<run_id>/random_forest_model'")
print("loaded_model = mlflow.sklearn.load_model(model_uri)")

# Get the latest run ID
client = mlflow.tracking.MlflowClient()
experiment_id = client.get_experiment_by_name("Telco Customer Churn Prediction").experiment_id
runs = client.search_runs(experiment_id)
if runs:
    latest_run_id = runs[0].info.run_id
    print(f"\nLatest run ID: {latest_run_id}")
    print(f"To load this model: model_uri = 'runs:/{latest_run_id}/random_forest_model'")

# %% [markdown]
# ## Making Predictions with the Model
# 
# Let's demonstrate how to use the model to make predictions on new data.

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
# 2. Load and preprocess the Telco Customer Churn dataset
# 3. Train multiple models (Decision Tree, Random Forest, XGBoost)
# 4. Track model parameters, metrics, and artifacts with MLflow
# 5. Save and load models using MLflow
# 6. Make predictions with the trained model
# 
# MLflow provides a powerful framework for managing the machine learning lifecycle, making it easier to track experiments, reproduce results, and deploy models. 
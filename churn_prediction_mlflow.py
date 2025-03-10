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

# Set MLflow tracking URI - this can be a local directory or a remote server
# For local tracking, we'll use a directory called 'mlruns'
mlflow.set_tracking_uri("file:./mlruns")

# Set the experiment name
mlflow.set_experiment("Telco Customer Churn Prediction")

# Load the CSV data to a pandas dataframe
data_path = "Data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(data_path)

# Data preprocessing
# Drop customer ID column
df = df.drop(columns=["customerID"])

# Handle missing values in TotalCharges
df["TotalCharges"] = df['TotalCharges'].replace({" ":"0.0"})
df["TotalCharges"] = df['TotalCharges'].astype(float)

# Label encoding for target column
df["Churn"] = df["Churn"].replace({"Yes":1, "No":0})

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

# Split the features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE for handling class imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Define models to train
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

# Train and log each model with MLflow
for model_name, model in models.items():
    # Start an MLflow run
    with mlflow.start_run(run_name=model_name):
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
        plt.close()
        
        # Log the confusion matrix plot
        mlflow.log_artifact(cm_plot_path)
        
        # Create feature importance plot (if available)
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            sns.barplot(x='Importance', y='Feature', data=feature_importance)
            plt.title(f'Feature Importance - {model_name}')
            
            # Save feature importance plot
            fi_plot_path = f"feature_importance_{model_name}.png"
            plt.savefig(fi_plot_path)
            plt.close()
            
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

print("\nMLflow tracking URI:", mlflow.get_tracking_uri())
print("Run 'mlflow ui' in the terminal to view the experiment results in a web UI")

# Example of loading a model from MLflow
print("\nExample of how to load a model from MLflow:")
print("model_uri = 'runs:/<run_id>/random_forest_model'")
print("loaded_model = mlflow.sklearn.load_model(model_uri)") 
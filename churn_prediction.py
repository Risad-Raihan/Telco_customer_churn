# -*- coding: utf-8 -*-
"""Churn Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1z3C0MgoKjzX5joO8AnMh9d1Yh9pY8n-K
"""

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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

"""2. **Data Loading and Understanding**


"""

# load teh csv data to a pandas dataframe
df = pd.read_csv("/content/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.shape

df

df.head()

df.info()

#droping customer id col
df = df.drop(columns=["customerID"])

df.head()

df.info()

# printing the unique values in all the columns

for col in df.columns:
  print(col, df[col].unique())
  print("-"*50)

#converting obj to numerical
#df["TotalCharges"] = df['TotalCharges'].astype(float)

len(df[df["TotalCharges"]==" "])

df[df["TotalCharges"]==" "]

df["TotalCharges"] = df['TotalCharges'].replace({" ":"0.0"})

df["TotalCharges"] = df['TotalCharges'].astype(float)

df.info()

#checking the claass distrubution of traget colum
print(df["Churn"].value_counts())



"""**Insights:**
1. we have removed customer id cz not required
2. no missing vales
3. missing values in TotalCahrges column were replavced with 0.0
4. Class imbalance identified in TG
"""



"""**3. EDA**"""

df.shape

df.columns

df.head(2)

df.describe()



"""**Numerical Features - Analysis**

UNderstand the distribution of Numerical feature
"""

def plot_histogram(df, column_name):

  plt.figure(figsize=(5, 3))
  sns.histplot(df[column_name], kde = True)
  plt.title(f"Distribution of {column_name}")

  #calcualte mean and median
  col_mean = df[column_name].mean()
  col_median = df[column_name].median()


  #add vertical lines for mean and median
  plt.axvline(col_mean, color = "red", linestyle = "--", label="Mean")
  plt.axvline(col_median, color = "green", linestyle = "-", label="Median")

  plt.legend()

  plt.show()

plot_histogram(df, "tenure")

plot_histogram(df, "MonthlyCharges")

plot_histogram(df, "TotalCharges")



"""**Bolx plot for numerical features**"""

def plot_boxplot(df, column_name):

  plt.figure(figsize=(5, 3))
  sns.boxplot(y=df[column_name])
  plt.title(f"Box Plot of {column_name}")
  plt.ylabel(column_name)
  plt.show

plot_boxplot(df, "tenure")

plot_boxplot(df, "MonthlyCharges")

plot_boxplot(df, "TotalCharges")



"""**correalation heatmap for numerical columns**"""

# correlation amtrix heatmap
plt.figure(figsize=(6, 3))
sns.heatmap(df[["tenure", "MonthlyCharges", "TotalCharges"]].corr(), annot=True, cmap = "coolwarm", fmt= ".2f")
plt.title("Correlation Heatmap")
plt.show()



"""**Categorical features - Analysis**"""

df.columns

"""Countplot for categorical columns"""

object_cols = df.select_dtypes(include="object").columns.to_list()

object_cols = ["SeniorCitizen"] + object_cols

for col in object_cols:
  plt.figure(figsize=(5, 3))
  sns.countplot(x=df[col])
  plt.title(f"Count Plot of {col}")
  plt.show()



"""4. Data Preprocessings"""

df.head(3)

"""Label encoding of tg column

"""

df["Churn"] = df["Churn"].replace({"Yes":1, "No":0})

df.head(3)

print(df["Churn"].value_counts())

"""Label encoding for categorical features"""

#identyfying columns iwth obeject dt
object_columns = df.select_dtypes(include="object").columns

print(object_columns)

#initialize a dictonary to save the encoders

encoders = {}

#apply label encoding and store the encoders

for column in object_columns:
  label_encoder = LabelEncoder()
  df[column] = label_encoder.fit_transform(df[column])
  encoders[column] = label_encoder

#save the encoders to a pickle file
with open("encoders.pkl", "wb") as f:
  pickle.dump(encoders, f)

encoders

df.head(3)



"""**Training and Test data split**"""

#split the features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

print(X)

#split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(y_train.shape)

print(y_train.value_counts())



"""**Synthetic Minority Oversampling Technique**"""

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(y_train_smote.shape)

print(y_train_smote.value_counts())

print(X_train_smote.shape)



"""5. Model Training

Training with default hyperparameters
"""

#dictionaries for model
models = {
      "Decision Tree": DecisionTreeClassifier(random_state=42),
      "Random Forest": RandomForestClassifier(random_state=42),
      "XGBoost": XGBClassifier(random_state=42)
}

# dictionary to store the cross validation results
cv_scores = {}

# perform 5-fold cross validation for each model
for model_name, model in models.items():
  print(f"Training {model_name} with default parameters")
  scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring="accuracy")
  cv_scores[model_name] = scores
  print(f"{model_name} cross-validation accuracy: {np.mean(scores):.2f}")
  print("-"*70)

cv_scores

rfc = RandomForestClassifier(random_state=42)

rfc.fit(X_train_smote, y_train_smote)

print(y_test.value_counts())

"""6. Model Evaluation"""

# evaluate on test  data
y_test_pred = rfc.predict(X_test)

print("Accuracy Score:\n", accuracy_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Accuracy Score:\n", classification_report(y_test, y_test_pred))

# save the trained model as a pickle file
model_data = {"model": rfc, "features_names": X.columns.tolist()}


with open("customer_churn_model.pkl", "wb") as f:
  pickle.dump(model_data, f)

"""7. Load the save model and build a predictive System"""

# load teh saved model and the feature names

with open("customer_churn_model.pkl", "rb") as f:
  model_data = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["features_names"]

print(loaded_model)

print(feature_names)

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


input_data_df = pd.DataFrame([input_data])

with open("encoders.pkl", "rb") as f:
  encoders = pickle.load(f)


# encode categorical featires using teh saved encoders
for column, encoder in encoders.items():
  input_data_df[column] = encoder.transform(input_data_df[column])

# make a prediction
prediction = loaded_model.predict(input_data_df)
pred_prob = loaded_model.predict_proba(input_data_df)

print(prediction)

# results
print(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
print(f"Prediciton Probability: {pred_prob}")


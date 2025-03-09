import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .prediction-text {
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
    }
    .churn {
        color: #D32F2F;
    }
    .no-churn {
        color: #388E3C;
    }
    .probability-gauge {
        margin-top: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load the model and encoders
@st.cache_resource
def load_model_and_encoders():
    try:
        with open("customer_churn_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        
        with open("encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
            
        return model_data, encoders
    except Exception as e:
        st.error(f"Error loading model or encoders: {e}")
        return None, None

model_data, encoders = load_model_and_encoders()

if model_data is not None and encoders is not None:
    loaded_model = model_data["model"]
    feature_names = model_data["features_names"]
    
    # Main header
    st.markdown("<h1 class='main-header'>Customer Churn Prediction</h1>", unsafe_allow_html=True)
    
    # App description
    st.markdown("""
    This application predicts whether a customer will churn (leave) or not based on various features.
    Fill in the customer information below and click 'Predict' to see the result.
    """)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Prediction", "Model Information", "Sample Data"])
    
    with tab1:
        st.markdown("<h2 class='sub-header'>Customer Information</h2>", unsafe_allow_html=True)
        
        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 1)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            
        with col2:
            multiple_lines = st.selectbox(
                "Multiple Lines", 
                ["No phone service", "No", "Yes"]
            )
            internet_service = st.selectbox(
                "Internet Service", 
                ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online Security", 
                ["No", "Yes", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", 
                ["No", "Yes", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection", 
                ["No", "Yes", "No internet service"]
            )
            
        with col3:
            tech_support = st.selectbox(
                "Tech Support", 
                ["No", "Yes", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", 
                ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", 
                ["No", "Yes", "No internet service"]
            )
            contract = st.selectbox(
                "Contract", 
                ["Month-to-month", "One year", "Two year"]
            )
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            
        # Payment method and charges in a new row
        col1, col2 = st.columns(2)
        
        with col1:
            payment_method = st.selectbox(
                "Payment Method", 
                [
                    "Electronic check", 
                    "Mailed check", 
                    "Bank transfer (automatic)", 
                    "Credit card (automatic)"
                ]
            )
            
        with col2:
            monthly_charges = st.number_input(
                "Monthly Charges ($)", 
                min_value=0.0, 
                max_value=500.0, 
                value=29.85,
                step=0.01
            )
            
            # Calculate a default for total charges based on tenure and monthly charges
            default_total = monthly_charges * tenure
            total_charges = st.number_input(
                "Total Charges ($)", 
                min_value=0.0, 
                max_value=10000.0, 
                value=default_total,
                step=0.01
            )
        
        # Create input data dictionary
        input_data = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Prediction button
        if st.button("Predict Churn"):
            # Convert input data to DataFrame
            input_data_df = pd.DataFrame([input_data])
            
            # Encode categorical features using the saved encoders
            for column, encoder in encoders.items():
                if column in input_data_df.columns:
                    input_data_df[column] = encoder.transform(input_data_df[column])
            
            # Make prediction
            prediction = loaded_model.predict(input_data_df)
            pred_prob = loaded_model.predict_proba(input_data_df)
            
            # Display prediction
            churn_status = "Churn" if prediction[0] == 1 else "No Churn"
            churn_class = "churn" if prediction[0] == 1 else "no-churn"
            
            st.markdown(f"""
            <div class='prediction-box' style='background-color: {"#FFEBEE" if prediction[0] == 1 else "#E8F5E9"}'>
                <p class='prediction-text {churn_class}'>Prediction: {churn_status}</p>
                <div class='probability-gauge'>
                    <p>Probability of churning: {pred_prob[0][1]:.2%}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a gauge chart for the probability
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh(["Churn Probability"], [pred_prob[0][1]], color='#FF6B6B')
            ax.barh(["Churn Probability"], [1-pred_prob[0][1]], left=[pred_prob[0][1]], color='#4CAF50')
            
            # Add a vertical line at the decision boundary (0.5)
            ax.axvline(x=0.5, color='black', linestyle='--')
            
            # Add text annotations
            ax.text(0.1, 0, "Low Risk", ha='center', va='center', color='white')
            ax.text(0.9, 0, "High Risk", ha='center', va='center', color='white')
            
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
            plt.tight_layout()
            st.pyplot(fig)
            
            # Feature importance (if available)
            if hasattr(loaded_model, 'feature_importances_'):
                st.subheader("Feature Importance")
                
                # Get feature importances
                importances = loaded_model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Plot feature importances
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.title('Feature Importances')
                plt.bar(range(len(indices)), importances[indices], color='b', align='center')
                plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
                plt.tight_layout()
                st.pyplot(fig)
    
    with tab2:
        st.markdown("<h2 class='sub-header'>Model Information</h2>", unsafe_allow_html=True)
        
        # Display model information
        st.write(f"**Model Type:** {type(loaded_model).__name__}")
        
        # Model parameters
        st.subheader("Model Parameters")
        st.json(loaded_model.get_params())
        
        # Feature list
        st.subheader("Features Used")
        st.write(feature_names)
    
    with tab3:
        st.markdown("<h2 class='sub-header'>Sample Data</h2>", unsafe_allow_html=True)
        
        # Load and display sample data
        try:
            sample_data = pd.read_csv("Data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
            st.dataframe(sample_data.head(10))
        except Exception as e:
            st.error(f"Error loading sample data: {e}")
            
else:
    st.error("Failed to load the model or encoders. Please check if the files exist and are in the correct format.") 
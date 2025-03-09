# Telco Customer Churn Prediction

![Churn Prediction](https://img.shields.io/badge/ML-Churn%20Prediction-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.43-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Overview

This project implements a machine learning model to predict customer churn for a telecommunications company. Customer churn, or customer attrition, is when customers stop using a company's services. Predicting churn helps businesses take proactive measures to retain customers.

The application features a user-friendly web interface built with Streamlit, allowing users to input customer information and receive real-time churn predictions.

## 🌟 Features

- **Interactive Web Interface**: Easy-to-use UI for inputting customer data
- **Real-time Predictions**: Instant churn probability assessment
- **Visual Results**: Graphical representation of prediction outcomes
- **Feature Importance**: Visualization of factors influencing churn
- **Model Information**: Details about the trained model
- **Sample Data Exploration**: View and analyze the dataset

## 🔍 Dataset

The model is trained on the Telco Customer Churn dataset, which includes information about:

- **Demographics**: Gender, senior citizen status, partners, dependents
- **Account Information**: Tenure, contract type, payment method, charges
- **Services**: Phone, internet, online security, tech support, streaming services

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and evaluation
- **XGBoost**: Gradient boosting framework
- **Matplotlib & Seaborn**: Data visualization
- **Streamlit**: Web application framework

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/Risad-Raihan/Telco_customer_churn.git
cd Telco_customer_churn
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL displayed in the terminal (usually http://localhost:8501)

3. Enter customer information in the form and click "Predict Churn" to see the prediction results

## 📊 Model Performance

The Random Forest model achieves:
- **Accuracy**: ~80%
- **Precision**: ~75%
- **Recall**: ~70%
- **F1 Score**: ~72%

## 📁 Project Structure

```
Telco_customer_churn/
├── app.py                      # Streamlit web application
├── churn_prediction.py         # Model training and evaluation code
├── Churn_Prediction.ipynb      # Jupyter notebook with analysis
├── customer_churn_model.pkl    # Trained machine learning model
├── encoders.pkl                # Label encoders for categorical features
├── requirements.txt            # Python dependencies
├── run.bat                     # Windows batch file to run the app
├── run.sh                      # Shell script to run the app
├── Data/                       # Dataset directory
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md                   # Project documentation
```

## 🔮 Future Improvements

- Implement more advanced models (Neural Networks, etc.)
- Add more visualization options
- Develop customer retention strategy recommendations
- Create a Docker container for easier deployment
- Add unit tests for model validation

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Risad Raihan**
- GitHub: [Risad-Raihan](https://github.com/Risad-Raihan)

## 🙏 Acknowledgments

- Dataset: IBM Sample Data Sets
- Built with Streamlit, Pandas, Scikit-learn, and XGBoost 
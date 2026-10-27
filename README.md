# Fraud_Detection_Project
Developed a Python-based fraud detection system analyzing 6.36 million transactions using machine learning and an interactive Streamlit dashboard. Strengthened skills in data analysis, pattern identification, and visualisation, applicable to sales trend analysis and demand planning.
FraudGuard – Financial Risk Intelligence

About the Project
FraudGuard is a machine learning-based fraud detection system that analyzes financial transactions to identify potentially fraudulent activity. It combines behavioral analysis, explainable AI, and an interactive dashboard to make fraud predictions easier to understand.

Dataset
Uses the "PaySim Fraud Detection Dataset" (https://www.kaggle.com/datasets/ealaxi/paysim1), containing approximately 6.36 million simulated financial transactions.

Key Features
- Detects potentially fraudulent transactions using machine learning.
- Creates behavioral features to identify suspicious patterns.
- Uses SHAP to explain the factors behind predictions.
- Provides an interactive dashboard for individual transactions and batch analysis.
- Categorizes estimated risk as High, Medium, or Low.

Technologies
Python, Pandas, NumPy, Scikit-learn, Random Forest, SHAP, Streamlit, and Matplotlib.

Model Performance
- ROC-AUC: 0.999833
- PR-AUC: 0.938557
- Fraud Recall: 95.25%
- Fraud Precision: Approximately 53%

Run the Dashboard
Install the required libraries:
pip install pandas numpy scikit-learn shap streamlit matplotlib joblib

Launch the application:
streamlit run src/fraud_dashboard.py

Disclaimer ⚠️
FraudGuard is an educational portfolio project built using simulated data. Its predictions indicate potential risk and should not be treated as proof of fraud.

import pandas as pd
import numpy as np
import joblib
import shap

# Load model and feature list
model = joblib.load(
    "models/behavioral_random_forest.pkl"
)

model_features = joblib.load(
    "models/model_features.pkl"
)

def prepare_transaction(transaction):

    df = pd.DataFrame([transaction])

    df["previous_transaction_count"] = transaction.get(
        "previous_transaction_count", 0
    )

    df["previous_amount"] = transaction.get(
        "previous_amount", 0
    )

    df["amount_vs_previous"] = (
        df["amount"] /
        (df["previous_amount"] + 1)
    )

    df["balance_change"] = (
        df["oldbalanceOrg"] -
        df["newbalanceOrig"]
    )

    df = pd.get_dummies(
        df,
        columns=["type"],
        dtype=int
    )

    df = df.reindex(
        columns=model_features,
        fill_value=0
    )

    return df


transaction = {
    "step": 400,
    "type": "TRANSFER",
    "amount": 50000,
    "oldbalanceOrg": 60000,
    "newbalanceOrig": 10000,
    "oldbalanceDest": 5000,
    "newbalanceDest": 55000,
    "isFlaggedFraud": 0,
    "previous_transaction_count": 5,
    "previous_amount": 10000
}

transaction_data = prepare_transaction(
    transaction
)

probability = model.predict_proba(
    transaction_data
)[0][1]

if probability >= 0.70:
    risk = "HIGH"
elif probability >= 0.30:
    risk = "MEDIUM"
else:
    risk = "LOW"

# Generate SHAP explanation
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    transaction_data
)

shap_fraud = shap_values[0, :, 1]

explanation = pd.DataFrame({
    "Feature": model_features,
    "SHAP Value": shap_fraud,
    "Absolute SHAP": np.abs(shap_fraud)
})

explanation = explanation.sort_values(
    "Absolute SHAP",
    ascending=False
)

top_reasons = explanation[
    explanation["SHAP Value"] > 0
].head(5)

print("\n========================================")
print("       FRAUD RISK ASSESSMENT")
print("========================================")

print(
    f"\nFraud Probability : {probability:.2%}"
)

print(
    f"Risk Level        : {risk}"
)

print("\nTop Fraud Indicators:")

for _, row in top_reasons.iterrows():
    print(
        f"- {row['Feature']} contributed "
        f"toward the fraud prediction."
    )

print("\n========================================")
import pandas as pd
import joblib

# Load saved model
model = joblib.load(
    "models/behavioral_random_forest.pkl"
)

model_features = joblib.load(
    "models/model_features.pkl"
)

def predict_fraud(transaction):

    df = pd.DataFrame([transaction])

    # Create engineered features
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

    # Encode transaction type
    df = pd.get_dummies(
        df,
        columns=["type"],
        dtype=int
    )

    # Match training features
    df = df.reindex(
        columns=model_features,
        fill_value=0
    )

    probability = model.predict_proba(
        df
    )[0][1]

    if probability >= 0.70:
        risk = "HIGH"
    elif probability >= 0.30:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return probability, risk


# Example transaction
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

probability, risk = predict_fraud(
    transaction
)

print("\n========================================")
print("       FRAUD RISK ASSESSMENT")
print("========================================")

print(
    f"\nFraud Probability : {probability:.2%}"
)

print(
    f"Risk Level        : {risk}"
)

print("\n========================================")
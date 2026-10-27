import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv(
    "dataset/PS_20174392719_1491204439457_log.csv"
)

# Create behavioral features
df = df.sort_values(["nameOrig", "step"])

df["previous_transaction_count"] = (
    df.groupby("nameOrig").cumcount()
)

df["previous_amount"] = (
    df.groupby("nameOrig")["amount"].shift(1)
)

df["amount_vs_previous"] = (
    df["amount"] / (df["previous_amount"] + 1)
)

df["balance_change"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)

features = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
    "previous_transaction_count",
    "previous_amount",
    "amount_vs_previous",
    "balance_change"
]

X = df[features].copy()
y = df["isFraud"].copy()

X["previous_amount"] = X["previous_amount"].fillna(0)
X["amount_vs_previous"] = X["amount_vs_previous"].fillna(1)

X = pd.get_dummies(
    X,
    columns=["type"],
    dtype=int
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Controlled training sample
fraud_train = X_train[y_train == 1]

normal_train = X_train[y_train == 0].sample(
    n=131400,
    random_state=42
)

X_sample = pd.concat([
    fraud_train,
    normal_train
])

y_sample = pd.concat([
    y_train.loc[fraud_train.index],
    y_train.loc[normal_train.index]
])

# Train final behavioral model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_sample, y_sample)

# Save model and feature names
joblib.dump(
    model,
    "models/behavioral_random_forest.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "models/model_features.pkl"
)

print("\nFinal model trained successfully!")

print("Model saved to:")
print("models/behavioral_random_forest.pkl")

print("\nFeature list saved to:")
print("models/model_features.pkl")
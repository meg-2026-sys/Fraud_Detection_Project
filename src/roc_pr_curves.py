import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, auc

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

# Select all fraud transactions
fraud_data = X_train[y_train == 1]
fraud_target = y_train[y_train == 1]

# Select normal transactions
normal_data = X_train[y_train == 0].sample(
    n=131400,
    random_state=42
)
normal_target = y_train.loc[normal_data.index]

# Create controlled training dataset
X_sample = pd.concat([
    fraud_data,
    normal_data
])

y_sample = pd.concat([
    fraud_target,
    normal_target
])

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_sample, y_sample)

print("\nModel trained successfully!")

# Generate fraud probabilities
y_probability = model.predict_proba(X_test)[:, 1]

# ROC Curve
fpr, tpr, _ = roc_curve(
    y_test,
    y_probability
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Behavioral Random Forest (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.title("ROC Curve - Behavioral Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()

plt.savefig(
    "graphs/behavioral_roc_curve.png",
    dpi=300
)

plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(
    y_test,
    y_probability
)

pr_auc = auc(
    recall,
    precision
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall,
    precision,
    label=f"Behavioral Random Forest (AUC = {pr_auc:.4f})"
)

plt.title("Precision-Recall Curve - Behavioral Random Forest")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()

plt.savefig(
    "graphs/behavioral_precision_recall_curve.png",
    dpi=300
)

plt.show()

print(f"\nROC-AUC: {roc_auc:.6f}")
print(f"PR-AUC: {pr_auc:.6f}")

print("\nROC and Precision-Recall graphs saved successfully!")
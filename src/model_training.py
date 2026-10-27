import pandas as pd
import matplotlib.pyplot as plt
import shap
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Find the project folder and dataset
project_folder = Path.cwd()
csv_files = list(project_folder.rglob("*.csv"))

if len(csv_files) == 0:
    print("ERROR: No CSV file found.")
else:
    file_path = csv_files[0]
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print("Dataset loaded!")
    print("Shape:", df.shape)


# Select the original transaction features
features = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud"
]

X = df[features]
y = df["isFraud"]

# Convert transaction type into numeric columns
X = pd.get_dummies(X, columns=["type"], dtype=int)

# Split the data while preserving the fraud ratio
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set:", X_train.shape)
print("Testing set:", X_test.shape)

print("\nTraining fraud distribution:")
print(y_train.value_counts())

print("\nTesting fraud distribution:")
print(y_test.value_counts())


# Create baseline models with class balancing
logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# Create a manageable training sample
train_data = pd.concat([X_train, y_train], axis=1)

fraud_data = train_data[
    train_data["isFraud"] == 1
]

normal_data = train_data[
    train_data["isFraud"] == 0
].sample(
    n=131400,
    random_state=42
)

train_sample = pd.concat([
    fraud_data,
    normal_data
])

X_train_sample = train_sample.drop(
    "isFraud",
    axis=1
)

y_train_sample = train_sample["isFraud"]

print("\nTraining sample:", X_train_sample.shape)
print("Fraud cases:", y_train_sample.sum())
print("Normal cases:", (y_train_sample == 0).sum())


# Train Logistic Regression
print("\nTraining Logistic Regression...")

logistic_model.fit(
    X_train_sample,
    y_train_sample
)

logistic_predictions = logistic_model.predict(X_test)
logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Results:")
print(
    classification_report(
        y_test,
        logistic_predictions
    )
)

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        logistic_probabilities
    )
)


# Train Random Forest
print("\nTraining Random Forest...")

random_forest_model.fit(
    X_train_sample,
    y_train_sample
)

rf_predictions = random_forest_model.predict(X_test)
rf_probabilities = random_forest_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest Results:")
print(
    classification_report(
        y_test,
        rf_predictions
    )
)

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        rf_probabilities
    )
)

print("\nConfusion Matrix - Random Forest:")
print(
    confusion_matrix(
        y_test,
        rf_predictions
    )
)


# Save the baseline model results
results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Precision": [
        0.03,
        0.52
    ],
    "Recall": [
        0.92,
        0.95
    ],
    "F1-Score": [
        0.06,
        0.67
    ],
    "ROC-AUC": [
        0.9874,
        0.9991
    ]
})

results.to_csv(
    project_folder /
    "results" /
    "baseline_results.csv",
    index=False
)

print("\nBaseline results saved.")


# Create the baseline comparison graph
results.set_index("Model")[
    [
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ]
].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Baseline Model Performance")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    project_folder /
    "graphs" /
    "baseline_model_comparison.png",
    dpi=300
)

plt.show()


# Create fast behavioral features
print("\nCreating behavioral features...")

df = df.sort_values(
    ["nameOrig", "step"]
)

# Count previous transactions by each customer
df["previous_transaction_count"] = (
    df.groupby("nameOrig").cumcount()
)

# Get the customer's previous transaction amount
df["previous_amount"] = (
    df.groupby("nameOrig")["amount"].shift(1)
)

# Compare current amount with the previous amount
df["amount_vs_previous"] = (
    df["amount"] /
    (df["previous_amount"] + 1)
)

# Calculate the change in the customer's original balance
df["balance_change"] = (
    df["oldbalanceOrg"] -
    df["newbalanceOrig"]
)

print("\nBehavioral features created!")

print(
    df[
        [
            "nameOrig",
            "amount",
            "previous_transaction_count",
            "previous_amount",
            "amount_vs_previous",
            "balance_change"
        ]
    ].head(10)
)

# Prepare the dataset for the behavioral model

behavioral_features = [
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

# Create the feature dataset
X_behavioral = df[behavioral_features].copy()

# Create the target separately
y_behavioral = df["isFraud"].copy()

# Replace missing previous-transaction values
X_behavioral["previous_amount"] = (
    X_behavioral["previous_amount"].fillna(0)
)

X_behavioral["amount_vs_previous"] = (
    X_behavioral["amount_vs_previous"].fillna(1)
)

# Convert transaction type into numeric columns
X_behavioral = pd.get_dummies(
    X_behavioral,
    columns=["type"],
    dtype=int
)

print("\nBehavioral dataset prepared!")
print("Shape:", X_behavioral.shape)

print("\nBehavioral features:")
print(X_behavioral.columns.tolist())

# Split the behavioral dataset
X_behavioral_train, X_behavioral_test, y_behavioral_train, y_behavioral_test = train_test_split(
    X_behavioral,
    y_behavioral,
    test_size=0.20,
    random_state=42,
    stratify=y_behavioral
)

print("\nBehavioral training set:", X_behavioral_train.shape)
print("Behavioral testing set:", X_behavioral_test.shape)

print("\nBehavioral training fraud distribution:")
print(y_behavioral_train.value_counts())

print("\nBehavioral testing fraud distribution:")
print(y_behavioral_test.value_counts())

# Create a manageable training sample
behavioral_train_data = pd.concat(
    [X_behavioral_train, y_behavioral_train],
    axis=1
)

behavioral_fraud = behavioral_train_data[
    behavioral_train_data["isFraud"] == 1
]

behavioral_normal = behavioral_train_data[
    behavioral_train_data["isFraud"] == 0
].sample(
    n=131400,
    random_state=42
)

behavioral_train_sample = pd.concat(
    [behavioral_fraud, behavioral_normal]
)

X_behavioral_sample = behavioral_train_sample.drop(
    "isFraud",
    axis=1
)

y_behavioral_sample = behavioral_train_sample["isFraud"]

print("\nBehavioral training sample:", X_behavioral_sample.shape)
print("Fraud cases:", y_behavioral_sample.sum())
print(
    "Normal cases:",
    (y_behavioral_sample == 0).sum()
)


# Create the behavioral Random Forest
behavioral_rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train the model
print("\nTraining Behavioral Random Forest...")

behavioral_rf.fit(
    X_behavioral_sample,
    y_behavioral_sample
)

# Generate predictions
behavioral_predictions = behavioral_rf.predict(
    X_behavioral_test
)

behavioral_probabilities = behavioral_rf.predict_proba(
    X_behavioral_test
)[:, 1]


# Evaluate the model
print("\nBehavioral Random Forest Results:")

print(
    classification_report(
        y_behavioral_test,
        behavioral_predictions
    )
)

behavioral_auc = roc_auc_score(
    y_behavioral_test,
    behavioral_probabilities
)

print("ROC-AUC:", behavioral_auc)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_behavioral_test,
        behavioral_predictions
    )
)

# Comparing model results
comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest - Baseline",
        "Random Forest - Behavioral"
    ],
    "Precision": [0.03, 0.52, 0.53],
    "Recall": [0.92, 0.95, 0.95],
    "F1 Score": [0.06, 0.67, 0.68],
    "ROC-AUC": [
        0.9874106578894408,
        0.9991261991037821,
        0.9998328092004927
    ]
})

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nModel comparison:")
print(comparison)

print("\nModel comparison saved successfully!")

# Creating model comparison graph
comparison_plot = comparison.set_index("Model")

comparison_plot[["Precision", "Recall", "F1 Score"]].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Fraud Detection Model Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()

plt.savefig(
    "graphs/model_comparison.png",
    dpi=300
)

plt.show()

print("\nComparison graph saved successfully!")

# Preparing a small sample for SHAP
shap_sample = X_behavioral_test.sample(
    n=500,
    random_state=42
)

print("\nSHAP sample shape:")
print(shap_sample.shape)

# Calculating SHAP explanations
explainer = shap.TreeExplainer(behavioral_rf)

shap_values = explainer.shap_values(shap_sample)

print("\nSHAP calculation completed!")
print("SHAP values type:", type(shap_values))

print("SHAP values shape:")
print(
    shap_values[1].shape
    if isinstance(shap_values, list)
    else shap_values.shape
)

# Extracting SHAP values for fraud class
shap_fraud = shap_values[:, :, 1]

shap_importance = np.abs(shap_fraud).mean(axis=0)

shap_importance_df = pd.DataFrame({
    "Feature": X_behavioral_test.columns,
    "SHAP Importance": shap_importance
}).sort_values(
    "SHAP Importance",
    ascending=False
)

print("\nTop SHAP features:")
print(shap_importance_df.head(10))

# Creating SHAP importance graph
plt.figure(figsize=(10, 6))

top_features = shap_importance_df.head(10).sort_values(
    "SHAP Importance"
)

plt.barh(
    top_features["Feature"],
    top_features["SHAP Importance"]
)

plt.title("Top Features Influencing Fraud Predictions")
plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    "graphs/shap_feature_importance.png",
    dpi=300
)

plt.show()

print("\nSHAP graph saved successfully!")

# Saving SHAP feature importance
shap_importance_df.to_csv(
    "results/shap_feature_importance.csv",
    index=False
)

print("\nSHAP feature importance saved successfully!")

# Selecting a fraud transaction for explanation
fraud_indices = y_behavioral_test[
    y_behavioral_test == 1
].index

fraud_transaction = X_behavioral_test.loc[
    fraud_indices[0]
]

fraud_transaction = fraud_transaction.to_frame().T

fraud_probability = behavioral_rf.predict_proba(
    fraud_transaction
)[0][1]

print("\nFraud transaction explanation:")
print("Fraud probability:", fraud_probability)

# Calculating SHAP values for the fraud transaction
fraud_shap = explainer.shap_values(
    fraud_transaction
)

fraud_shap = fraud_shap[0, :, 1]

fraud_explanation = pd.DataFrame({
    "Feature": X_behavioral_test.columns,
    "SHAP Value": fraud_shap,
    "Absolute SHAP": np.abs(fraud_shap)
}).sort_values(
    "Absolute SHAP",
    ascending=False
)

print("\nTop factors for this fraud prediction:")
print(fraud_explanation.head(10))

# Creating individual fraud explanation graph
top_factors = fraud_explanation.head(10).sort_values(
    "SHAP Value"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_factors["Feature"],
    top_factors["SHAP Value"]
)

plt.axvline(
    x=0,
    linewidth=1
)

plt.title(
    "SHAP Explanation for a Fraudulent Transaction"
)

plt.xlabel("SHAP Value")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    "graphs/individual_fraud_explanation.png",
    dpi=300
)

plt.show()

print("\nIndividual fraud explanation graph saved successfully!")

# Creating fraud risk assessment

def get_risk_level(probability):
    if probability >= 0.70:
        return "HIGH"
    elif probability >= 0.30:
        return "MEDIUM"
    else:
        return "LOW"


risk_level = get_risk_level(fraud_probability)

print("\nFraud Risk Assessment:")
print("-----------------------------")
print(f"Fraud Probability : {fraud_probability:.2%}")
print(f"Risk Level        : {risk_level}")
print("-----------------------------")

# Generating human-readable explanations

top_reasons = fraud_explanation[
    fraud_explanation["SHAP Value"] > 0
].head(5)

print("\nTop Fraud Indicators:")

for _, row in top_reasons.iterrows():
    print(
        f"- {row['Feature']} contributed "
        f"toward the fraud prediction."
    )

# Saving the fraud explanation

fraud_result = pd.DataFrame({
    "Fraud Probability": [fraud_probability],
    "Risk Level": [risk_level],
    "Top Feature 1": [fraud_explanation.iloc[0]["Feature"]],
    "Top Feature 2": [fraud_explanation.iloc[1]["Feature"]],
    "Top Feature 3": [fraud_explanation.iloc[2]["Feature"]],
    "Top Feature 4": [fraud_explanation.iloc[3]["Feature"]],
    "Top Feature 5": [fraud_explanation.iloc[4]["Feature"]]
})

fraud_result.to_csv(
    "results/example_fraud_explanation.csv",
    index=False
)

print("\nFraud explanation saved successfully!")
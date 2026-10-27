import pandas as pd

# Load final model results
df = pd.read_csv(
    "results/final_model_results.csv"
)

baseline = df[
    df["Model"] == "Random Forest - Baseline"
].iloc[0]

behavioral = df[
    df["Model"] == "Random Forest - Behavioral"
].iloc[0]

# Calculate improvements
precision_change = behavioral["Precision"] - baseline["Precision"]
recall_change = behavioral["Recall"] - baseline["Recall"]
f1_change = behavioral["F1 Score"] - baseline["F1 Score"]
roc_change = behavioral["ROC-AUC"] - baseline["ROC-AUC"]

print("\nBehavioral Model Improvement")
print("=" * 50)

print(
    f"Precision improvement : {precision_change:.4f}"
)

print(
    f"Recall improvement    : {recall_change:.4f}"
)

print(
    f"F1 improvement        : {f1_change:.4f}"
)

print(
    f"ROC-AUC improvement   : {roc_change:.6f}"
)

# Calculate percentage improvement
precision_percent = (
    precision_change / baseline["Precision"]
) * 100

f1_percent = (
    f1_change / baseline["F1 Score"]
) * 100

roc_percent = (
    roc_change / baseline["ROC-AUC"]
) * 100

print("\nPercentage Improvement")
print("=" * 50)

print(
    f"Precision : {precision_percent:.2f}%"
)

print(
    f"F1 Score  : {f1_percent:.2f}%"
)

print(
    f"ROC-AUC   : {roc_percent:.4f}%"
)
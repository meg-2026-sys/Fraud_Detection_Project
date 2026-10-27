import pandas as pd

# Behavioral Random Forest confusion matrix
true_negative = 1269505
false_positive = 1376
false_negative = 78
true_positive = 1565

total = (
    true_negative
    + false_positive
    + false_negative
    + true_positive
)

print("\nFraud Detection Error Analysis")
print("=" * 50)

print(f"True Negatives  : {true_negative:,}")
print(f"False Positives : {false_positive:,}")
print(f"False Negatives : {false_negative:,}")
print(f"True Positives  : {true_positive:,}")

print(f"\nTotal Test Transactions: {total:,}")

# Error rates
false_positive_rate = (
    false_positive /
    (false_positive + true_negative)
)

false_negative_rate = (
    false_negative /
    (false_negative + true_positive)
)

print("\nError Rates")
print("=" * 50)

print(
    f"False Positive Rate : {false_positive_rate:.4%}"
)

print(
    f"False Negative Rate : {false_negative_rate:.4%}"
)

# Save results
results = pd.DataFrame({
    "Metric": [
        "True Negatives",
        "False Positives",
        "False Negatives",
        "True Positives",
        "False Positive Rate",
        "False Negative Rate"
    ],
    "Value": [
        true_negative,
        false_positive,
        false_negative,
        true_positive,
        false_positive_rate,
        false_negative_rate
    ]
})

results.to_csv(
    "results/error_analysis.csv",
    index=False
)

print("\nError analysis saved successfully!")
print("results/error_analysis.csv")
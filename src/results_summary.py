import pandas as pd

# Final model comparison
results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest - Baseline",
        "Random Forest - Behavioral"
    ],
    "Precision": [
        0.03,
        0.52,
        0.53
    ],
    "Recall": [
        0.92,
        0.95,
        0.95
    ],
    "F1 Score": [
        0.06,
        0.67,
        0.68
    ],
    "ROC-AUC": [
        0.987410658,
        0.999126199,
        0.999832809
    ]
})

# Save results
results.to_csv(
    "results/final_model_results.csv",
    index=False
)

print("\nFinal Model Results")
print("=" * 60)
print(results.to_string(index=False))

print("\nResults saved successfully!")
print("results/final_model_results.csv")
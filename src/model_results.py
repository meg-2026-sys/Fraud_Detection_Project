import pandas as pd
import matplotlib.pyplot as plt

# Load saved model comparison results
results = pd.read_csv("results/model_comparison.csv")

print("\nModel Results:")
print(results)

# Plot performance comparison
metrics = ["Precision", "Recall", "F1 Score", "ROC-AUC"]

ax = results.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Fraud Detection Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.legend(title="Metrics")
plt.tight_layout()

plt.savefig(
    "graphs/final_model_performance.png",
    dpi=300
)

plt.show()

print("\nFinal model performance graph saved successfully!")
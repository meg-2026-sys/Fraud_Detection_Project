import pandas as pd
import matplotlib.pyplot as plt

# Load final model results
results = pd.read_csv("results/final_model_results.csv")

metrics = ["Precision", "Recall", "F1 Score", "ROC-AUC"]

ax = results.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Final Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.legend(title="Metrics")
plt.tight_layout()

plt.savefig(
    "graphs/final_metrics_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Final metrics graph saved successfully!")
print("graphs/final_metrics_performance.png")
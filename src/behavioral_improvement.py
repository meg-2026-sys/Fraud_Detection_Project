import pandas as pd
import matplotlib.pyplot as plt

baseline = pd.DataFrame({
    "Metric": [
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Baseline Random Forest": [
        0.52,
        0.95,
        0.67,
        0.999126
    ],
    "Behavioral Random Forest": [
        0.53,
        0.95,
        0.68,
        0.999833
    ]
})

ax = baseline.set_index("Metric").plot(
    kind="bar",
    figsize=(9, 5)
)

ax.set_ylabel("Score")
ax.set_title(
    "Baseline vs Behavioral Random Forest"
)

ax.set_ylim(0, 1.05)

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "graphs/behavioral_improvement.png",
    dpi=300
)

plt.show()
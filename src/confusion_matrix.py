import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Actual behavioral model confusion matrix
cm = np.array([
    [1269505, 1376],
    [78, 1565]
])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Fraud"]
)

disp.plot()
plt.title("Confusion Matrix - Behavioral Random Forest")
plt.tight_layout()

plt.savefig(
    "graphs/behavioral_confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix graph saved successfully!")
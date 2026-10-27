import pandas as pd
import matplotlib.pyplot as plt

# SHAP importance values from the behavioral model
shap_results = pd.DataFrame({
    "Feature": [
        "balance_change",
        "newbalanceOrig",
        "oldbalanceOrg",
        "amount",
        "type_PAYMENT",
        "newbalanceDest",
        "type_TRANSFER",
        "oldbalanceDest",
        "step",
        "type_CASH_OUT"
    ],
    "SHAP Importance": [
        0.142681,
        0.084067,
        0.081083,
        0.061264,
        0.046575,
        0.041244,
        0.038952,
        0.032895,
        0.025415,
        0.022958
    ]
})

# Sort for visualization
shap_results = shap_results.sort_values(
    "SHAP Importance"
)

plt.figure(figsize=(9, 6))

plt.barh(
    shap_results["Feature"],
    shap_results["SHAP Importance"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title(
    "Top Features Influencing Fraud Predictions"
)

plt.tight_layout()

plt.savefig(
    "graphs/shap_feature_importance.png",
    dpi=300
)

plt.show()

# Save table
shap_results.to_csv(
    "results/shap_feature_importance.csv",
    index=False
)

print("\nSHAP summary created successfully!")
print("Graph: graphs/shap_feature_importance.png")
print("Table: results/shap_feature_importance.csv")
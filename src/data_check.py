import pandas as pd
from pathlib import Path

# Find the project folder
project_folder = Path.cwd()

print("Project folder:")
print(project_folder)

print("\nSearching for CSV files...")

# Search for CSV files inside the project folder
csv_files = list(project_folder.rglob("*.csv"))

print("\nCSV files found:")

for file in csv_files:
    print(file)

# Check if a CSV was found
if len(csv_files) == 0:
    print("\nERROR: No CSV file was found.")
else:

# Use the first CSV file found
    file_path = csv_files[0]

    print("\nLoading:")
    print(file_path)

    df = pd.read_csv(file_path)

    print("\nDataset loaded successfully!")

    print("\nNumber of rows and columns:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 transactions:")
    print(df.head())

    print("\nFraud distribution:")
    print(df["isFraud"].value_counts())

    print("\nFraud percentage:")
    print(df["isFraud"].value_counts(normalize=True) * 100)

# Fraud rate by transaction type
    fraud_by_type = df.groupby("type")["isFraud"].mean() * 100

    print("\nFraud percentage by transaction type:")
    print(fraud_by_type.sort_values(ascending=False))

    import matplotlib.pyplot as plt

# Create fraud-rate graph
    plt.figure(figsize=(8, 5))

    fraud_by_type.sort_values(ascending=False).plot(kind="bar")

    plt.title("Fraud Rate by Transaction Type")
    plt.xlabel("Transaction Type")
    plt.ylabel("Fraud Rate (%)")
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()


print("\nCreating faster behavioral features...")

# Sort transactions by customer and time
df = df.sort_values(["nameOrig", "step"])

# Previous transaction count
df["prev_transaction_count"] = df.groupby("nameOrig").cumcount()

# Previous transaction amount
df["prev_amount"] = df.groupby("nameOrig")["amount"].shift(1)

# Previous transaction amount ratio
df["amount_vs_prev"] = df["amount"] / (df["prev_amount"] + 1)

# Balance change
df["balance_change"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)

print("\nBehavioral features created successfully!")

print(df[[
    "nameOrig",
    "step",
    "amount",
    "prev_transaction_count",
    "prev_amount",
    "amount_vs_prev",
    "balance_change",
    "isFraud"
]].head(15))

# Behavioral feature analysis
print("\nBehavioral feature analysis:")

# Average ratio for normal vs fraud
print("\nAverage amount compared with previous transaction:")
print(
    df.groupby("isFraud")["amount_vs_prev"]
      .mean()
)

# Median ratio for normal vs fraud
print("\nMedian amount compared with previous transaction:")
print(
    df.groupby("isFraud")["amount_vs_prev"]
      .median()
)

# Store median values for the graph
median_behavior = df.groupby("isFraud")["amount_vs_prev"].median()

# Create bar chart
plt.figure(figsize=(7, 5))
median_behavior.plot(kind="bar")

plt.title("Median Transaction Amount Ratio: Normal vs Fraud")
plt.xlabel("Transaction Status")
plt.ylabel("Median Amount vs Previous Transaction")
plt.xticks([0, 1], ["Normal", "Fraud"], rotation=0)

plt.tight_layout()
plt.show()
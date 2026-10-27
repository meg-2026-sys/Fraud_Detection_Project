import pandas as pd
from pathlib import Path

project_folder = Path.cwd()

csv_files = list(project_folder.rglob("*.csv"))

dataset_path = [
    file for file in csv_files
    if "PS_20174392719_1491204439457_log" in file.name
][0]

df = pd.read_csv(dataset_path)

normal = df[df["isFraud"] == 0].sample(
    10,
    random_state=42
)

fraud = df[df["isFraud"] == 1].sample(
    10,
    random_state=42
)

test_batch = pd.concat(
    [normal, fraud]
).sample(
    frac=1,
    random_state=42
)

output_path = (
    project_folder /
    "results" /
    "test_batch_transactions.csv"
)

test_batch.to_csv(
    output_path,
    index=False
)

print("Test batch created successfully!")
print(f"Transactions: {len(test_batch)}")
print(f"Normal: {(test_batch['isFraud'] == 0).sum()}")
print(f"Fraud: {(test_batch['isFraud'] == 1).sum()}")
print(f"\nSaved to:\n{output_path}")
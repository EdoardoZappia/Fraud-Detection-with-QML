import pandas as pd

# Specify the paths to your split datasets
half1_path = "../dataset/half1.csv"
half2_path = "../dataset/half2.csv"

# Read the split datasets into DataFrames using Pandas
try:
    half1 = pd.read_csv(half1_path)
    half2 = pd.read_csv(half2_path)
    print("Split datasets imported successfully.")
except FileNotFoundError:
    print("One or both split datasets not found. Please check the file paths.")
    exit()

# Merge the two halves into a single dataset
try:
    full_dataset = pd.concat([half1, half2], ignore_index=True)
    print("Split datasets merged into a single dataset successfully.")
    print("Shape of the merged dataset:", full_dataset.shape)
except Exception as e:
    print("An error occurred while merging the datasets:", e)

try:
    full_dataset.to_csv("../dataset/merged_dataset.csv", index=False)
    print("Merged dataset saved as merged_dataset.csv")
except Exception as e:
    print("An error occurred while saving the merged dataset:", e)
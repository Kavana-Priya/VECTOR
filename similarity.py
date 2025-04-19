#!/usr/bin/python
#
# This file is part of VECTOR.
# Contribution: Kavana Priyadarshini Keshava,Dieter W. Heermann, Arnab Bhattacherjee 
# Usage :python similarity.py ./celline1/clustered_entropy.csv ./cellline2/clustered_entropy.csv 
###################Pakages####################################################
import argparse
import pandas as pd
from scipy.stats import pearsonr

parser = argparse.ArgumentParser(description='Compare Entropy values between two CSV files')
parser.add_argument('file1', type=str, help='Path to the first CSV file')
parser.add_argument('file2', type=str, help='Path to the second CSV file')

args = parser.parse_args()

# Load both CSVs
df1 = pd.read_csv(args.file1)
df2 = pd.read_csv(args.file2)

# Rename Entropy_Scaled columns for clarity
df1 = df1.rename(columns={"Entropy_Scaled": "Entropy_Scaled_1"})
df2 = df2.rename(columns={"Entropy_Scaled": "Entropy_Scaled_2"})

# Merge on x_values with inner join (only keep common x_values)
merged_df = pd.merge(df1[["x_values", "Entropy_Scaled_1"]], 
                     df2[["x_values", "Entropy_Scaled_2"]], 
                     on="x_values", how="inner")

# Calculate Pearson correlation
corr, p_value = pearsonr(merged_df["Entropy_Scaled_1"], merged_df["Entropy_Scaled_2"])

# Print results
print("Pearson Correlation Coefficient:", corr)

# Optional: Save merged data
merged_df.to_csv("merged_entropy.csv", index=False)


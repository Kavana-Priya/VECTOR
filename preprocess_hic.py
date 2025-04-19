#!/usr/bin/python
#
# This file is part of VECTOR.
# Contribution: Kavana Priyadarshini Keshava,Dieter W. Heermann, Arnab Bhattacherjee 
# Usage: python preprocess_hic.py file.hic chromosome:start:end resolution --normalization KR/NONE
#############################################Packages################################################

import hicstraw
import pandas as pd
import numpy as np
import argparse
import sys

# -------------------------------
# Step 1: Parse command-line args
# -------------------------------
parser = argparse.ArgumentParser(description='Extract symmetric Hi-C matrix from .hic file')
parser.add_argument('hic_file', type=str, help='Path to the .hic file')
parser.add_argument('chromosome', type=str, help='Chromosome region (e.g., "2:60000000:110000000")')
parser.add_argument('resolution', type=int, help='Resolution (e.g., 50000)')
parser.add_argument('--normalization', type=str, choices=['KR', 'VC', 'NONE'], default='KR',
                    help='Normalization method (default: KR)')

args = parser.parse_args()

print(f"\n Loading Hi-C data from: {args.hic_file}")
print(f"Region: {args.chromosome}")
print(f"Resolution: {args.resolution} bp")
print(f"Normalization: {args.normalization}")

# -----------------------------
# Step 2: Load data with hicstraw
# -----------------------------
try:
    result = hicstraw.straw('observed', args.normalization, args.hic_file,
                             args.chromosome, args.chromosome, 'BP', args.resolution)
    if not result:
        print("No Hi-C contacts found for the specified region. Exiting.")
        sys.exit(1)
except Exception as e:
    print(f"Error reading Hi-C data: {e}")
    sys.exit(1)

print(f"Loaded {len(result)} contact records")

# --------------------------------------
# Step 3: Convert to DataFrame (sparse)
# --------------------------------------
data = [(r.binX, r.binY, r.counts) for r in result]
sparse_df = pd.DataFrame(data, columns=['binX', 'binY', 'counts'])

print("Sparse DataFrame created:")
print(sparse_df.head())

# ----------------------------------
# Step 4: Make symmetric matrix
# ----------------------------------
mirrored_df = sparse_df.copy()
mirrored_df.columns = ['binY', 'binX', 'counts']  # flip binX and binY

symmetric_df = pd.concat([sparse_df, mirrored_df], ignore_index=True)
symmetric_df = symmetric_df.groupby(['binX', 'binY'], as_index=False).sum()

# -------------------------------
# Step 5: Save symmetric sparse
# -------------------------------
if not symmetric_df.empty:
    symmetric_df.to_csv('symmetric_sparse_hic.tsv', index=False, sep='\t')
    print("Saved symmetric sparse matrix to 'symmetric_sparse_hic.tsv'")
else:
    print("symmetric_df is empty, skipping file save.")

# ---------------------------------------
# Step 6: Create dense matrix from sparse
# ---------------------------------------
bins = sorted(set(symmetric_df['binX']).union(set(symmetric_df['binY'])))
bin_to_index = {b: i for i, b in enumerate(bins)}
n = len(bins)
dense_matrix = np.zeros((n, n))

for _, row in symmetric_df.iterrows():
    i = bin_to_index[row['binX']]
    j = bin_to_index[row['binY']]
    dense_matrix[i, j] = row['counts']

# -----------------------------------------------
# Step 7: Save dense 3-column matrix (binX, binY)
# -----------------------------------------------
dense_triplets = []
for i in range(n):
    for j in range(n):
        count = dense_matrix[i, j]
        if count != 0:
            dense_triplets.append((bins[i], bins[j], count))

dense_triplet_df = pd.DataFrame(dense_triplets, columns=['binX', 'binY', 'counts'])

if not dense_triplet_df.empty:
    dense_triplet_df.to_csv('symmetric_dense_3col.tsv', header=False, index=False, sep='\t')
    print("Saved dense 3-column matrix to 'symmetric_dense_3col.tsv'")
else:
    print("Dense matrix is empty, skipping file save.")

print("Done.")


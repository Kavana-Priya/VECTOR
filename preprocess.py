import hicstraw
import pandas as pd
import numpy as np

# Step 1: Load Hi-C data
result = hicstraw.straw('observed', 'KR', '4DNFICXCFGEI.hic',
                        '2:60000000:110000000', '2:60000000:110000000', 'BP', 50000)

# Step 2: Convert to sparse DataFrame
data = [(r.binX, r.binY, r.counts) for r in result]
sparse_df = pd.DataFrame(data, columns=['binX', 'binY', 'counts'])

# Step 3: Make symmetric by mirroring
mirrored_df = sparse_df.copy()
mirrored_df.columns = ['binY', 'binX', 'counts']  # Swap binX and binY

# Combine original and mirrored, then group to sum duplicates
symmetric_df = pd.concat([sparse_df, mirrored_df], ignore_index=True)
symmetric_df = symmetric_df.groupby(['binX', 'binY'], as_index=False).sum()

# Step 4: Save symmetric sparse matrix
symmetric_df.to_csv('symmetric_sparse_hic.tsv', index=False, sep='\t')

# Step 5: Create dense matrix
bins = sorted(set(symmetric_df['binX']).union(set(symmetric_df['binY'])))
bin_to_index = {b: i for i, b in enumerate(bins)}
n = len(bins)
dense_matrix = np.zeros((n, n))

for _, row in symmetric_df.iterrows():
    i = bin_to_index[row['binX']]
    j = bin_to_index[row['binY']]
    dense_matrix[i, j] = row['counts']

# Step 6: Save dense matrix in 3-column format (binX, binY, counts)
dense_triplets = []
for i in range(n):
    for j in range(n):
        count = dense_matrix[i, j]
        if count != 0:
            dense_triplets.append((bins[i], bins[j], count))

dense_triplet_df = pd.DataFrame(dense_triplets, columns=['binX', 'binY', 'counts'])
dense_triplet_df.to_csv('symmetric_dense_3col.tsv', index=False, sep='\t')


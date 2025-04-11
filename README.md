# VECTOR - Von nEumann entropy deteCTiOn of stRuctural patterns in Hi-C data

VECTOR is a computational tool designed to quantify and compare Hi-C contact maps using a graph-based approach. It enables the differentiation of two Hi-C matrices — including those from different developmental stages or even biological replicates — by transforming contact frequency data into graph representations.

Each Hi-C matrix is modeled as a graph where genomic loci are nodes and contact frequencies are weighted edges. Using this transformation, VECTOR computes the Laplacian matrix and derives its eigenvalues to calculate the Von Neumann entropy, which serves as a measure of the complexity or structural diversity within the contact map.

This method provides a robust and sensitive metric to evaluate subtle changes in chromatin organization across conditions, developmental stages, or experimental replicates. 

# Requirements

PYTHON

numpy  
networkx  
matplotlib  
scikit-learn


# Usage

1. Download .hic File
Obtain Hi-C data in .hic format from a public database such as the 4D Nucleome Data Portal or GEO.

2. Extract Region Using straw
Use Juicebox's straw tool to extract a specific genomic region from the .hic file into a sparse matrix format.Ensure your matrix is in a format readable by the entropy.py script (e.g., tab-separated).

`    python preprocess.py
`

4. Run Entropy Analysis
Run the Python script to compute Von Neumann entropy from the matrix:

`    python VECTOR.py
`
5. Output

The entropy values are saved to a .csv file for downstream analysis or visualization.



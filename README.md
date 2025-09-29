# VECTOR - Von nEumann entropy deteCTiOn of stRuctural patterns in Hi-C data

VECTOR is a computational tool designed to quantify and compare Hi-C contact maps using a graph-based approach. It enables the differentiation of two Hi-C matrices - including those from different developmental stages or even biological replicates - by transforming contact frequency data into graph representations.

The Hi-C matrix is modeled as a graph where genomic loci are nodes and contact frequencies are weighted edges. Using this transformation, VECTOR computes the Laplacian matrix and derives its eigenvalues to calculate the Von Neumann entropy, which serves as a measure of the complexity or structural diversity within the contact map.

This method provides a robust and sensitive metric to evaluate subtle changes in chromatin organization across conditions, developmental stages, or experimental replicates. 

# Requirements

PYTHON

numpy  
networkx  
matplotlib  
scikit-learn
hicstraw


# Usage

1. Download .hic File
Obtain Hi-C data in .hic format from a public database such as the 4D Nucleome Data Portal or GEO.

2. Extract Genomic Region Using straw
Use the straw tool from Juicebox to extract a specific genomic region from the .hic file into a tab-separated 3 column format compatible with the VECTOR.py script.

```
python preprocess_hic.py file.hic chromosome:start:end resolution --normalization KR/NONE

```

3. Run VECTOR : Entropy Analysis
Compute the Von Neumann entropy of the extracted Hi-C contact matrix using the VECTOR.py script.

```
python VECTOR.py symmetric_dense_3col.tsv

```
This will output entropy values saved in a .csv file.


4. Entropy Similarity Check

Compare Von Neumann entropy profiles between two Hi-C datasets using the similarity.py script:

```
python similarity.py ./celline1/clustered_entropy.csv ./cellline2/clustered_entropy.csv 

```   
This step helps quantify structural similarity based on entropy features.




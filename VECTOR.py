#!/usr/bin/python
#
# This file is part of VECTOR.
# Contribution: Kavana Priyadarshini Keshava,Dieter W. Heermann, Arnab Bhattacherjee 
# Usage : python VECTOR.py symmetric_dense_3col.tsv
##############################Packages####################################################
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.colors as mcolors
import csv
import pandas as pd
import argparse
from numpy import linalg as LA
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
##########################################################################################
# HELPER FUNCTIONS
##########################################################################################


def read_matrix_new(filename,G):


  data = []
  entries = set()
  consistency = {}
  with open(filename, 'r') as f:
      for line in f:
          if not line.strip():
            continue
          row = [float(x) for x in line.strip().split()]
          if int(row[0]) not in entries:
              entries.add(int(row[0]))
          if int(row[1]) not in entries:
              entries.add(int(row[1]))

          data.append([row[0], row[1], row[2]])
          consistency[(int(row[0]),int(row[1]))] = row[2]

  c_keys = consistency.keys()
  # Consistency check
  for k in c_keys:
      k0 = k[0]
      k1 = k[1]
      if (k1,k0) not in c_keys:
          print('not in: ',k1,k0)


  #print('len entries = ',len(entries))
  key_list = sorted(list(entries))
  #print(key_list[:10])
  mx = max(key_list)
  mi = min(key_list)
  delta = list(set([x - key_list[i - 1] for i, x in enumerate(key_list)][1:]))
  #print(mi,mx,delta)

  idx = list(range(mi,mx+min(delta),min(delta)))
  mapping = {k:v for v,k in enumerate(idx)}
  # Save mapping to CSV
  with open('mapping.csv', 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(['Key', 'Value'])
      writer.writerows(mapping.items())
  print('Mapping saved to mapping.csv')
  #print(mapping)
  n = len(idx)
  #print(n)
  
  
  matrix = np.zeros((n,n))
  for d in data:
      i = mapping[int(d[0])]
      j = mapping[int(d[1])]
      matrix[i,j] = d[2]
      #matrix[j,i] = d[2]
  
  x = matrix.sum(axis=1) 
  for i in range(len(x)):
      if x[i] == 0:
          #print('zero',i)
          matrix[i,i] = 0.001
          #print(matrix[:,i])

  G = nx.from_numpy_array(matrix,edge_attr='weight')

  return G, matrix,mapping


def data_to_matrix(data,mapping):

    dim = len(mapping.keys())
    #print(dim)

    matrix = np.zeros((dim,dim))
    

    for tup in data:
        a,b,w = tup[0],tup[1],tup[2]
        i = mapping[a]
        j = mapping[b]
        matrix[i][j] = np.log(w)

    return matrix


def read_matrix_mapping(filename):


  data = []
  mapping = {}
  i = 0
  with open(filename, 'r') as f:
    for line in f:
      if not line.strip():
        continue
      row = [float(x) for x in line.strip().split()]
      if float(row[2]) > 0 and row[0] != row[1]:
          if int(row[0]) not in mapping.keys():
              mapping[int(row[0])] = i
              i += 1
          data.append([row[0], row[1], row[2]])


  return data,mapping

def is_invertible(matrix):
    return np.linalg.cond(matrix) < 1 / np.finfo(matrix.dtype).eps


def plot_part(G,file):


    fig, ax = plt.subplots(figsize=(20.0,20.0))

    pos = nx.nx_agraph.graphviz_layout(G, prog='neato')

    # Plot network
    nx.draw_networkx(G, pos,
                     font_size=16,
                     alpha=0.5,
                     width=5.0,linewidths=0.8,
                     edge_color='gray',
                     node_color='tab:blue',
                     node_shape='o',
                     node_size=5000,
                     ax=ax)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('part_' + file + '.pdf',transparent=True)
    plt.close()


def plot_heatmap_ax_eigen(ax,matrix,mapping,y_pred):

    from matplotlib.colors import LinearSegmentedColormap
    
    colors = mcolors.TABLEAU_COLORS
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    names = list(colors)  

    # Define a red-white colormap (linear interpolation)
    red = plt.cm.RdPu(0)  # Pure red
    white = plt.cm.RdPu(1)  # Pure white

    # Define the color gradient from white to bright red
    cmap_colors = [(1, 1, 1), (1, 0.9, 0.9), (1, 0.7, 0.7), (1, 0.5, 0.5), (1, 0.2, 0.2)]  # White to bright red gradient
    cmap_positions = [0, 0.25, 0.5, 0.75, 1]  # Position of each color in the gradient


    # Define the color gradient from white to bright red
    cmap_colors = [(1, 1, 1), (1, 0.9, 0.9), (1,0.8,0.8), (1, 0.7, 0.7),(1, 0.6, 0.6), 
                   (1, 0.5, 0.5),(1, 0.4, 0.4), (1, 0.3, 0.3), (1, 0.2, 0.2), (1, 0.1, 0.1)]  # White to bright red gradient
    cmap_positions = [0, 0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1]  # Position of each color in the gradient


    # Create the custom colormap
    cmap_name = 'white_to_red'
    cmap = LinearSegmentedColormap.from_list(cmap_name, list(zip(cmap_positions, cmap_colors)))
  
    #heatmap = ax.imshow(matrix, norm='log', cmap='hot')  # 'hot' is a colormap, choose another if desired
 
    heatmap = ax.imshow(matrix[:-1, :-1], cmap=cmap, norm='log')  # 'hot' is a colormap, choose another if desired
    
    for i in range(matrix.shape[0]):
        #    #ax.axvline(x=mapping[float(v)],c=v, alpha=0.6,lw=6)
        ax.axvline(x=i,c=names[y_pred[i]], alpha=0.1,lw=1)
        ax.axhline(y=i,c=names[y_pred[i]], alpha=0.1,lw=1)

    ax.set_xlabel("Column Labels")
    ax.set_ylabel("Row Labels")
    plt.axis('off')


def plot_heatmap_ax(ax,matrix,mapping,y_pred):

    from matplotlib.colors import LinearSegmentedColormap
    
    colors = mcolors.TABLEAU_COLORS
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    names = list(colors)  

    # Define a red-white colormap (linear interpolation)
    red = plt.cm.RdPu(0)  # Pure red
    white = plt.cm.RdPu(1)  # Pure white


    # Define the color gradient from white to bright red
    cmap_colors = [(1, 1, 1), (1, 0.9, 0.9), (1,0.8,0.8), (1, 0.7, 0.7),(1, 0.6, 0.6), 
                   (1, 0.5, 0.5),(1, 0.4, 0.4), (1, 0.3, 0.3), (1, 0.2, 0.2), (1, 0.1, 0.1)]  # White to bright red gradient
    cmap_positions = [0, 0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1]  # Position of each color in the gradient


    # Create the custom colormap
    cmap_name = 'white_to_red'
    cmap = LinearSegmentedColormap.from_list(cmap_name, list(zip(cmap_positions, cmap_colors)))
  
    #heatmap = ax.imshow(matrix, norm='log', cmap='hot')  # 'hot' is a colormap, choose another if desired
 
    heatmap = ax.imshow(matrix[:-1, :-1], cmap=cmap, norm='log')  # 'hot' is a colormap, choose another if desired
    plt.title("stride=5")
    
    plt.axis('off')



def visualize_clusters(data, labels, method='pca'):
    if method == 'pca':
        # Reduce the dimensionality to 2D using PCA
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        # Reduce the dimensionality to 2D using t-SNE
        reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    else:
        raise ValueError("Unknown method: {}".format(method))
    
    reduced_data = reducer.fit_transform(data)
    
    # Plot the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis',alpha=0.3)
    plt.title(f"Spectral Clustering Results with {method.upper()}")
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.colorbar(label='Cluster')

    plt.savefig('cluster_' + method + '.pdf')
    #plt.show()
    plt.close()



##########################################################################################
# FUNCTIONS
##########################################################################################


def von_neumann_graph_entropy_weighted(graph):
    """
    Compute the von Neumann entropy of a weighted graph based on its normalized Laplacian matrix.

    Args:
        graph (networkx.Graph): The input weighted graph.

    Returns:
        float: The von Neumann entropy of the graph.
    """
    # Construct the weighted adjacency matrix
    adj_matrix = nx.to_numpy_array(graph, weight='weight')
    n = adj_matrix.shape[0]

    # Construct the weighted degree matrix
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    if not is_invertible(degree_matrix):
        return -1

    # Compute the combinatorial Laplacian matrix
    laplacian = degree_matrix - adj_matrix

    # Compute the normalized Laplacian matrix
    degree_sqrt = np.sqrt(np.linalg.inv(degree_matrix))
    normalized_laplacian = np.eye(n) - degree_sqrt @ adj_matrix @ degree_sqrt

    # Compute the eigenvalues of the normalized Laplacian
    eigenvalues = np.linalg.eigvalsh(normalized_laplacian)

    # Treat the non-zero eigenvalues as a probability distribution
    non_zero_eigenvalues = eigenvalues[eigenvalues > 0]
    probabilities = non_zero_eigenvalues / np.sum(non_zero_eigenvalues)
    
    # Compute the von Neumann entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


def generate_random_graph(n, p):
    return nx.erdos_renyi_graph(n, p)


def test_random_graph(n, p):
    graph = generate_random_graph(n, p)
    entropy = von_neumann_graph_entropy_weighted(graph)
    #print(f"Random Graph von Neumann entropy: {entropy:.4f}")


def generate_sub_graph_(d,delta,matrix):

    submatrix = np.array(matrix[d-delta:d+delta,d-delta:d+delta], copy=True)
    submatrix[range(submatrix.shape[0]), range(submatrix.shape[0])] = 0

    # Set the upper off-diagonals to zero, starting from the second diagonal
    submatrix[np.triu_indices_from(submatrix, k=delta//2+1)] = 0

    # Set the lower off-diagonals to zero, starting from the second diagonal
    submatrix[np.tril_indices_from(submatrix, k=-delta//2-1)] = 0


    for i in range(0,delta+delta//2):
        for j in range(0,delta//2):
            submatrix[i,j] = 0
            submatrix[i+delta//2,j+delta + delta//2] = 0


    #submatrix = np.array(matrix[d:d+delta,d:d+delta], copy=True)
    #submatrix[range(submatrix.shape[0]), range(submatrix.shape[0])] = 0

    G = nx.from_numpy_array(submatrix)
    G.remove_edges_from([(u, v) for u, v in G.edges() if u == v])        
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)

    return G.subgraph(Gcc[0])


def analyze_matrix(matrix):

    '''
    The principal eigenvector (corresponding to the largest positive eigenvalue) 
    gives a measure of vertex centrality or importance in the graph. The components of 
    this eigenvector indicate the relative importance or centrality of each vertex. 
    Vertices with larger components are more central or important in the graph structure.

    Graph partitioning and communities: The higher eigenvectors (after the principal e
    igenvector) can reveal insights into the community structure or partitioning of the 
    graph. The signs and magnitudes of the components in these eigenvectors can be 
    used to identify clusters or communities of vertices that are densely connected 
    within themselves but sparsely connected to other parts of the graph.
    '''

    eigenvalues, eigenvectors = LA.eigh(matrix)

    n = matrix.shape[0]

    fig, ax = plt.subplots(nrows=3,ncols=2,figsize=(11.69,11.69))

    ax[0,0].plot(range(n),eigenvectors[-1], label='EigenVec One')
    ax[0,0].legend()
    ax[0,0].set_title('Adjacency')

    ax[1,0].plot(range(n),eigenvectors[-2], label='Scnd Larg EigenVec')
    ax[1,0].legend()

    data = eigenvectors[-2].tolist()

    ax[2,0].hist(data, label='Hist EV of Scnd Larg EV (cluster structure)')
    ax[2,0].legend()



    '''
    LAPLACIAN:
    
    The eigenvectors corresponding to higher eigenvalues (especially the second smallest 
    eigenvalue) reveal information about the graph's connectivity and clustering properties.
    
    The entries of these eigenvectors can be used to partition the vertices into clusters 
    or communities, where vertices with similar values tend to belong to the same cluster.

    The sign and magnitude of the entries in these eigenvectors indicate the strength of 
    association between vertices and their respective clusters.
    '''

    # Construct the weighted degree matrix
    degree_matrix = np.diag(np.sum(matrix, axis=1))
    if not is_invertible(degree_matrix):
        return -1

    # Compute the combinatorial Laplacian matrix    
    laplacian = degree_matrix - matrix

    # Compute the normalized Laplacian matrix
    degree_sqrt = np.sqrt(np.linalg.inv(degree_matrix))
    normalized_laplacian = np.eye(n) - degree_sqrt @ matrix @ degree_sqrt

    eigenvalues, eigenvectors = LA.eigh(normalized_laplacian)
    # Use the first `num_clusters` eigenvectors (excluding the first smallest eigenvalue which is zero)


    ax[0,1].plot(range(n),eigenvectors[-1], label='EV One')
    ax[0,1].legend()
    ax[0,1].set_title('Laplacian')

    ax[1,1].plot(range(n),eigenvectors[-2], label='Scnd Largest EV')
    ax[1,1].legend()

    data = eigenvectors[-2].tolist()

    ax[2,1].hist(data, label='Hist EV of Scnd Larg EV (cluster structure)')
    ax[2,1].legend()

    plt.savefig('eigen_analysis.pdf')

    #plt.show()
    plt.close()


def analyze_eigenvectors(matrix,mapping,num_clusters):
    '''
    Spectral Clustering
    
    The eigenvectors of the normalized Laplacian are often used in spectral clustering 
    algorithms, which aim to partition the graph into clusters or communities based on 
    the eigenvector information.

    These algorithms typically use the top few eigenvectors (excluding the constant 
    eigenvector) to embed the vertices in a lower-dimensional space, where clusters can 
    be more easily identified and separated.
    '''

    n = matrix.shape[0]
    
    # Construct the weighted degree matrix
    degree_matrix = np.diag(np.sum(matrix, axis=1))
    if not is_invertible(degree_matrix):
        return -1


    # Compute the combinatorial Laplacian matrix    
    laplacian = degree_matrix - matrix

    # Compute the normalized Laplacian matrix
    degree_sqrt = np.sqrt(np.linalg.inv(degree_matrix))
    normalized_laplacian = np.eye(n) - degree_sqrt @ matrix @ degree_sqrt

    eigenvalues, eigenvectors = LA.eigh(normalized_laplacian)


    # Use the first `num_clusters` eigenvectors (excluding the first smallest eigenvalue which is zero)
    embedding = eigenvectors[:, 1:num_clusters+1]
    
    # Use k-means to partition the graph into clusters
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(embedding)
    #print(clusters)


    # Plot mapping to heatmap
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(11.69,11.69))

    plot_heatmap_ax_eigen(ax,matrix,mapping,clusters)
    
    plt.savefig('eigen_map_' + '.pdf')
    plt.close()


    visualize_clusters(embedding, clusters,method='tsne')

    return clusters


def analyze_graph_entropy(G,stride,num_clusters):


    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(11.69,11.69))

    # Set image extent based on your image data shape
    n = matrix.shape[0]
    print(n)
    image_extent = [0, n, 0, n]


    # Define a function to transform y-axis data to image coordinates
    def y_to_image_coord(y):
        # Map y-data to image height range based on your y-axis scale
        image_height = image_extent[3] - image_extent[2]
        y_norm = (y - min(y_data)) / (max(y_data) - min(y_data))
        return image_extent[2] + y_norm * image_height


    ents = []
    x = []
    for i in range(0,n-1,stride):

        G_p = nx.ego_graph(G, i, radius=1)
     
        #plot_part(G_p,'_graph_' + str(i))        
        n = G_p.number_of_nodes()
        entropy_p = von_neumann_graph_entropy_weighted(G_p)
        #print(i,n,entropy_p/np.log2(n))
        
        if entropy_p > 0 :
            x.append(i)
            ents.append((entropy_p/np.log2(n)))

            for eval in range(0,stride,1):
                with open('mapping.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([i+eval,n, entropy_p / np.log2(n)])

    # Transform y-data to image coordinates
    image_height = image_extent[3] - image_extent[2]
    y_norm = (ents - min(ents)) / (max(ents) - min(ents))
    y_image_coord = image_extent[2] + y_norm * image_height
    lw1=3
    #ax.plot(x,y_image_coord,lw=lw1)
    #ax.boxplot(x,y_image_coord)

    num_clusters = 4

    # Use k-means to partition the graph into clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0,n_init=10).fit(np.asarray(ents).reshape(-1, 1))
    clusters = kmeans.labels_

    #print(clusters)

    colors = mcolors.TABLEAU_COLORS
    names = list(colors)
    ents1 = np.asarray(ents).reshape(-1, 1)  # Ensure ents is a 1D numpy array

    # Calculate the mean entropy for each cluster
    mean_entropies = np.array([np.mean(ents1[clusters == i]) for i in range(num_clusters)])

# Sort clusters based on mean entropy in descending order
    sorted_cluster_indices = np.argsort(-mean_entropies)

# Create a mapping from old cluster IDs to new sorted cluster IDs
    cluster_id_mapping = np.zeros_like(clusters)
    for new_id, old_id in enumerate(sorted_cluster_indices):
        cluster_id_mapping[clusters == old_id] = new_id

    #print(clusters)
    clusters_sorted = cluster_id_mapping
    #print("Original Clusters:", clusters)
    #print("Sorted Clusters:", clusters_sorted)

    colors = mcolors.TABLEAU_COLORS
    names = list(colors)
# Create boxplot of clusters
    plt.figure(figsize=(10, 6))
    colors = list(mcolors.TABLEAU_COLORS.values())  # Get colors from Tableau palette

# Boxplot of entropies grouped by cluster
    for i in range(num_clusters):
        cluster_data = y_norm[cluster_id_mapping == i]
        plt.boxplot(cluster_data, positions=[i], patch_artist=True,
                    boxprops=dict(facecolor=colors[i], color=colors[i]),
                    whiskerprops=dict(color=colors[i]),
                    capprops=dict(color=colors[i]),
                    medianprops=dict(color="black"))

# Customize the plot
    plt.xlabel("Cluster ID")
    plt.ylabel("Entropy")
    plt.title("Boxplot of Entropies by Cluster")
    plt.xticks(range(num_clusters), [f'Cluster {i}' for i in range(num_clusters)])
    plt.ylim(0,1)
    plt.show()

    colors = mcolors.TABLEAU_COLORS
    names = list(colors)

    d = 0
    for i in x:
        with open('cluster.csv', 'a', newline='') as file:
            writer = csv.writer(file,delimiter='\t')
            writer.writerow([i,clusters[d],ents[d]])
            #print(ents[d])
            ax.axvline(x=i,c=names[clusters[d]], alpha=0.4,lw=2)
            ax.axhline(y=i,c=names[clusters[d]], alpha=0.4,lw=2)
            d += 1

    plot_heatmap_ax(ax,matrix,mapping,clusters_sorted)


    fig.savefig('graph_entropy_' +'.pdf')

def remove_outliers_iqr(df, column, multiplier=1):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


##########################################################################################
# MAIN
##########################################################################################


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description="Load graph from dense 3-column file")
    parser.add_argument('file', type=str, help='Path to the 3-column TSV file')
    args = parser.parse_args()
    G = nx.Graph()
    file1=args.file
    G, matrix, mapping = read_matrix_new(file1,G)
    G.remove_edges_from([(u, v) for u, v in G.edges() if u == v])        


    num_clusters = 20
    clusters = analyze_eigenvectors(matrix,mapping,num_clusters)    
    analyze_matrix(matrix)

    
    num_clusters = 4
    stride = 10
    analyze_graph_entropy(G,stride,num_clusters)
    file_path = "cluster.csv"  # Update with actual file path
    df = pd.read_csv(file_path, sep="\t", header=None, names=["x_values", "col2", "Entropy_Scaled"])
    df = df.drop(columns=["col2"])
    # Remove outliers from 'A549' column
    df_cleaned = remove_outliers_iqr(df, "Entropy_Scaled").reset_index(drop=True)
    # Scale entropy values using MinMaxScaler
    scaler = MinMaxScaler()
    df_cleaned["Entropy_Scaled"] = scaler.fit_transform(df_cleaned[["Entropy_Scaled"]])

    # Perform KMeans clustering
    num_clusters = 4
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df_cleaned["cluster"] = kmeans.fit_predict(df_cleaned[["Entropy_Scaled"]])

    cluster_order = np.argsort(kmeans.cluster_centers_.reshape(-1))
    cluster_map = {old: new + 1 for new, old in enumerate(cluster_order)}
    df_cleaned["cluster"] = df_cleaned["cluster"].map(cluster_map)

    df_sorted = df_cleaned.sort_values(by=["x_values"]).reset_index(drop=True)

    df_sorted.to_csv("clustered_entropy.csv", index=False)

    print(df_sorted.head())


    exit()

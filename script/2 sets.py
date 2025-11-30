import numpy as np
import os
import matplotlib.pyplot as plt
from read_data import *

# 1. CONFIGURATION
# ---------------------------------------------------------
# Change this to your dataset folder name
DATASET_NAME = '1000_20'
NUM_CLUSTERS = 1000

print(f"Working Directory: {os.getcwd()}")
print(f"Target Dataset: {DATASET_NAME}")

# Updated Paths
dir_partitions = f'..\\data\\sift\\{DATASET_NAME}\\partitions'
dir_ground = f'..\\data\\sift\\sift_groundtruth.ivecs'
dir_query = f'..\\data\\sift\\sift_query.fvecs'
dir_stats = f'..\\data\\sift\\{DATASET_NAME}\\cluster_stats'

os.makedirs(dir_stats, exist_ok=True)

# 2. LOAD GLOBAL DATA
# ---------------------------------------------------------
g = inspect_file_data(dir_ground)
q = inspect_file_data(dir_query)

# 3. COMPUTE/LOAD CLUSTER STATISTICS
# ---------------------------------------------------------
centroids = []
all_ids = []
effective_radii = []
principal_components = []
eigenvalues_list = []
cluster_stds = []

print(f"Computing statistics for {NUM_CLUSTERS} clusters...")

for it in range(NUM_CLUSTERS):
    # Print progress every 100 clusters
    if it % 100 == 0: print(f"  Processing cluster {it}...", end='\r')

    # Construct paths
    path_vecs = os.path.join(dir_partitions, f'partition_{it}.fvecs')
    path_ids = os.path.join(dir_partitions, f'partition_{it}.ids')

    # Load data
    v = inspect_file_data(path_vecs)
    i = inspect_file_data(path_ids)

    # --- Basic Stats ---
    centroid = np.mean(v, axis=0)
    centroids.append(centroid)
    all_ids.append(set(i))

    # Effective Radius (Mean distance)
    dists = np.linalg.norm(v - centroid, axis=1)
    radius = np.mean(dists)
    effective_radii.append(radius)

    # Per-dimension STD (Diagonal Covariance Proxy)
    std_per_dim = np.std(v, axis=0)
    cluster_stds.append(std_per_dim)

    # --- PCA (Full Covariance) ---
    centered = v - centroid
    # PCA: eig(Covariance)
    # Note: rowvar=False means columns are variables (dims)
    cov_matrix = np.cov(centered, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

    # Sort descending (largest variance first)
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    principal_components.append(eig_vecs)
    eigenvalues_list.append(eig_vals)

    # Save to disk (optional, caches stats)
    np.save(os.path.join(dir_stats, f'centroid_{it}.npy'), centroid)
    np.save(os.path.join(dir_stats, f'pca_{it}.npy'), eig_vecs)
    np.save(os.path.join(dir_stats, f'eigenvalues_{it}.npy'), eig_vals)

print(f"\nStats computed. Mean Radius: {np.mean(effective_radii):.2f}")

centroids = np.array(centroids)
effective_radii = np.array(effective_radii)
cluster_stds = np.array(cluster_stds)


# 4. METRIC DEFINITIONS
# ---------------------------------------------------------
def euclidean_to_centroid(query, centroids):
    return np.linalg.norm(centroids - query, axis=1)


def normalized_gap(query, centroids, effective_radii):
    """Euclidean Distance / Cluster Radius"""
    dists = np.linalg.norm(centroids - query, axis=1)
    return dists / (effective_radii + 1e-9)


def projected_mahalanobis(query, centroids, pcs, eig_vals, k=10):
    """
    Project query vector onto the cluster's principal components.
    Measures distance weighted by standard deviations in those directions.
    """
    scores = []
    for i in range(len(centroids)):
        diff = query - centroids[i]

        # Project onto top K components
        # (D,) dot (D, K) -> (K,)
        proj = np.dot(diff, pcs[i][:, :k])

        # Mahalanobis: divide by std dev (sqrt of eigenvalue)
        # We add epsilon to avoid division by zero
        sigmas = np.sqrt(eig_vals[i][:k]) + 1e-9
        normalized_proj = proj / sigmas

        # Norm of the normalized vector
        dist = np.linalg.norm(normalized_proj)
        scores.append(dist)
    return np.array(scores)


def boundary_heuristic(query, centroids, stds):
    """
    Simple heuristic: Distance / Per-Dimension STD
    Faster than full PCA but captures axis-aligned stretching.
    """
    scores = []
    for i in range(len(centroids)):
        diff = np.abs(query - centroids[i])
        # How many stds away is it on average across dimensions?
        z_scores = diff / (stds[i] + 1e-9)
        scores.append(np.mean(z_scores))
    return np.array(scores)


# 5. EVALUATION LOOP
# ---------------------------------------------------------
methods = {
    'Euclidean': lambda q: euclidean_to_centroid(q, centroids),
    'Normalized Gap': lambda q: normalized_gap(q, centroids, effective_radii),
    'Projected (k=10)': lambda q: projected_mahalanobis(q, centroids, principal_components, eigenvalues_list, k=10),
    'Boundary (Simple)': lambda q: boundary_heuristic(q, centroids, cluster_stds)
}

num_test = 100
k_clusters = 10  # We check the top 10 clusters selected by the method
k_recall = 100  # We look for the top 100 GT neighbors

results = {m: {'recall': [], 'overlap': []} for m in methods}
oracle_recalls = []

print(f"\nEvaluated {num_test} queries...")

for qid in range(num_test):
    query = q[qid]
    gt = set(g[qid][:k_recall])  # Top 100 GT

    # Oracle: Which clusters actually contain the GT points?
    actual_counts = []
    for cid in range(NUM_CLUSTERS):
        count = len(gt & all_ids[cid])
        actual_counts.append((cid, count))

    # Sort by count descending
    actual_counts.sort(key=lambda x: x[1], reverse=True)
    best_clusters = set(cid for cid, count in actual_counts[:k_clusters])

    # Oracle Recall (Upper Bound)
    oracle_ids = set()
    for cid in best_clusters:
        oracle_ids |= all_ids[cid]
    oracle_recalls.append(len(gt & oracle_ids) / k_recall)

    # Evaluate Methods
    for name, func in methods.items():
        scores = func(query)
        # Lower score is better (distance)
        ranking = np.argsort(scores)
        selected = set(ranking[:k_clusters])

        # 1. Overlap with Oracle (How well did we pick the 'right' clusters?)
        overlap = len(selected & best_clusters)
        results[name]['overlap'].append(overlap)

        # 2. Actual Recall
        ids_found = set()
        for cid in selected:
            ids_found |= all_ids[cid]
        rec = len(gt & ids_found) / k_recall
        results[name]['recall'].append(rec)

# 6. RESULTS
# ---------------------------------------------------------
print("\n" + "=" * 70)
print(f"{'Method':<20} {'Overlap':<10} {'Recall@100':<15} {'Gap to Oracle'}")
print("=" * 70)

oracle_avg = np.mean(oracle_recalls) * 100

for name in methods:
    ov = np.mean(results[name]['overlap'])
    rec = np.mean(results[name]['recall']) * 100
    gap = oracle_avg - rec
    print(f"{name:<20} {ov:<10.2f} {rec:<15.2f}% {gap:.2f}%")

print(f"{'Oracle (Upper Bound)':<20} {10:<10.2f} {oracle_avg:<15.2f}% 0.00%")
print("=" * 70)

# 7. PLOTTING
# ---------------------------------------------------------




fig, ax = plt.subplots(figsize=(10, 6))
data = [results[m]['recall'] for m in methods]
data.append(oracle_recalls)
labels = list(methods.keys()) + ['Oracle']

# Multiply by 100 for percentage
data = [np.array(d) * 100 for d in data]

ax.boxplot(data, labels=labels, patch_artist=True,
           boxprops=dict(facecolor='lightblue', color='blue'),
           medianprops=dict(color='red'))

ax.set_title(f'Recall@100 Distribution ({NUM_CLUSTERS} Clusters, {k_clusters} Visited)')
ax.set_ylabel('Recall (%)')
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(f'metrics_comparison_{DATASET_NAME}.png')
plt.show()
print(f"Plot saved to metrics_comparison_{DATASET_NAME}.png")
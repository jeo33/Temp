import numpy as np
import os
from read_data import *
import matplotlib.pyplot as plt

print(os.getcwd())
dir = f'..\\data\\sift\\1000_20\\partitions'
groud_dir = f'..\\data\\sift\\sift_groundtruth.ivecs'
base = f'..\\data\\sift\\sift_base.fvecs'
query_path = f'..\\data\\sift\\sift_query.fvecs'
stats_dir = f'..\\data\\sift\\1000_20\\cluster_stats'

os.makedirs(stats_dir, exist_ok=True)

g = inspect_file_data(groud_dir)
b = inspect_file_data(base)
q = inspect_file_data(query_path)

# Load all partitions and compute statistics
centroids = []
all_ids = []
effective_radii = []
principal_components = []  # top eigenvectors per cluster
eigenvalues_list = []
cluster_stds = []  # per-dimension std

print("Computing cluster statistics...")
for it in range(1000):
    vecs = os.path.join(dir, f'partition_{it}.fvecs')
    ids = os.path.join(dir, f'partition_{it}.ids')
    v = inspect_file_data(vecs)
    i = inspect_file_data(ids)

    centroid = np.mean(v, axis=0)
    centroids.append(centroid)
    all_ids.append(set(i))

    # Per-cluster effective radius
    distances_to_centroid = np.linalg.norm(v - centroid, axis=1)
    effective_radius = np.mean(distances_to_centroid)
    effective_radii.append(effective_radius)

    # Per-dimension standard deviation
    std_per_dim = np.std(v, axis=0)
    cluster_stds.append(std_per_dim)

    # PCA: get top-k principal components
    centered = v - centroid
    cov_matrix = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    principal_components.append(eigenvectors)
    eigenvalues_list.append(eigenvalues)

    # Save
    np.save(os.path.join(stats_dir, f'centroid_{it}.npy'), centroid)
    np.save(os.path.join(stats_dir, f'radius_{it}.npy'), effective_radius)
    np.save(os.path.join(stats_dir, f'pca_{it}.npy'), eigenvectors)
    np.save(os.path.join(stats_dir, f'eigenvalues_{it}.npy'), eigenvalues)
    np.save(os.path.join(stats_dir, f'std_{it}.npy'), std_per_dim)

centroids = np.array(centroids)
effective_radii = np.array(effective_radii)
cluster_stds = np.array(cluster_stds)

print(
    f"Effective radius - Mean: {np.mean(effective_radii):.2f}, Min: {np.min(effective_radii):.2f}, Max: {np.max(effective_radii):.2f}")


# Distance functions
def euclidean_to_centroid(query, centroids):
    return np.linalg.norm(centroids - query, axis=1)


def gap_per_cluster_radius(query, centroids, effective_radii):
    """Gap using each cluster's own radius"""
    centroid_dists = np.linalg.norm(centroids - query, axis=1)
    gaps = centroid_dists - effective_radii
    return gaps


def normalized_gap(query, centroids, effective_radii):
    """Distance normalized by cluster radius: dist / radius"""
    centroid_dists = np.linalg.norm(centroids - query, axis=1)
    return centroid_dists / (effective_radii + 1e-10)


def projected_distance(query, centroids, principal_components, eigenvalues_list, k_dims=20):
    """Project query onto each cluster's principal components and compute weighted distance"""
    distances = []
    for i in range(len(centroids)):
        diff = query - centroids[i]
        # Project onto top-k principal components
        pc = principal_components[i][:, :k_dims]
        projected = np.dot(diff, pc)
        # Weight by inverse sqrt of eigenvalue (larger spread = less important)
        weights = 1.0 / np.sqrt(eigenvalues_list[i][:k_dims] + 1e-6)
        weighted_dist = np.sqrt(np.sum((projected * weights) ** 2))
        distances.append(weighted_dist)
    return np.array(distances)


def boundary_distance(query, centroids, cluster_stds):
    """Estimate distance to cluster boundary using per-dimension std"""
    distances = []
    for i in range(len(centroids)):
        diff = np.abs(query - centroids[i])
        # How many stds away in each dimension
        z_scores = diff / (cluster_stds[i] + 1e-10)
        # Use mean or max z-score
        distances.append(np.mean(z_scores))
    return np.array(distances)


def hybrid_score(query, centroids, effective_radii, cluster_stds, alpha=0.5):
    """Combine normalized gap with boundary distance"""
    norm_gap = normalized_gap(query, centroids, effective_radii)
    bound_dist = boundary_distance(query, centroids, cluster_stds)
    # Normalize each
    norm_gap = (norm_gap - np.mean(norm_gap)) / (np.std(norm_gap) + 1e-10)
    bound_dist = (bound_dist - np.mean(bound_dist)) / (np.std(bound_dist) + 1e-10)
    return alpha * norm_gap + (1 - alpha) * bound_dist


# Test
num_test_queries = 100
k_clusters = 10
k_neighbors = 100

methods = {
    'euclidean': lambda q: euclidean_to_centroid(q, centroids),
    'gap_per_radius': lambda q: gap_per_cluster_radius(q, centroids, effective_radii),
    'normalized_gap': lambda q: normalized_gap(q, centroids, effective_radii),
    'projected_k20': lambda q: projected_distance(q, centroids, principal_components, eigenvalues_list, k_dims=20),
    'projected_k10': lambda q: projected_distance(q, centroids, principal_components, eigenvalues_list, k_dims=10),
    'boundary': lambda q: boundary_distance(q, centroids, cluster_stds),
    'hybrid': lambda q: hybrid_score(q, centroids, effective_radii, cluster_stds),
}

results = {name: {'overlaps': [], 'recalls': []} for name in methods}
results['oracle'] = {'recalls': []}

print(f"\nTesting {num_test_queries} queries...")
for qid in range(num_test_queries):
    query = q[qid]
    gt_neighbors = set(g[qid])

    # Oracle
    gt_counts = np.array([len(gt_neighbors & all_ids[cid]) for cid in range(1000)])
    best_actual_clusters = set(np.argsort(gt_counts)[::-1][:k_clusters])

    ids_in_oracle = set()
    for cid in best_actual_clusters:
        ids_in_oracle |= all_ids[cid]
    oracle_recall = len(gt_neighbors & ids_in_oracle) / k_neighbors
    results['oracle']['recalls'].append(oracle_recall)

    # Test each method
    for name, dist_func in methods.items():
        dists = dist_func(query)
        selected = set(np.argsort(dists)[:k_clusters])
        overlap = len(selected & best_actual_clusters)

        ids_in_selected = set()
        for cid in selected:
            ids_in_selected |= all_ids[cid]
        recall = len(gt_neighbors & ids_in_selected) / k_neighbors

        results[name]['overlaps'].append(overlap)
        results[name]['recalls'].append(recall)

# Summary
print("\n" + "=" * 75)
print(f"{'Method':<18} {'Overlap':<12} {'Recall':<12} {'Gap to Oracle':<15}")
print("=" * 75)

oracle_mean = np.mean(results['oracle']['recalls']) * 100

for name in methods:
    overlap = np.mean(results[name]['overlaps'])
    recall = np.mean(results[name]['recalls']) * 100
    gap = oracle_mean - recall
    print(f"{name:<18} {overlap:<12.2f} {recall:<11.1f}% {gap:<14.1f}%")

print(f"{'oracle':<18} {10:<12} {oracle_mean:<11.1f}% {0:<14.1f}%")
print("=" * 75)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

method_names = list(methods.keys())
colors = plt.cm.tab10(np.linspace(0, 1, len(method_names)))

# Recall boxplot
ax1 = axes[0]
recall_data = [np.array(results[m]['recalls']) * 100 for m in method_names]
recall_data.append(np.array(results['oracle']['recalls']) * 100)
bp1 = ax1.boxplot(recall_data, tick_labels=method_names + ['oracle'], patch_artist=True)
for patch, color in zip(bp1['boxes'], list(colors) + [(0.5, 0.5, 0.5, 1)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.set_ylabel('Recall@100 (%)')
ax1.set_title('Recall by Method')
ax1.tick_params(axis='x', rotation=45)

# Overlap boxplot
ax2 = axes[1]
overlap_data = [results[m]['overlaps'] for m in method_names]
bp2 = ax2.boxplot(overlap_data, tick_labels=method_names, patch_artist=True)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_ylabel('Cluster Overlap with Oracle (out of 10)')
ax2.set_title('Cluster Selection Accuracy')
ax2.tick_params(axis='x', rotation=45)
ax2.axhline(10, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('per_cluster_radius_comparison.png', dpi=150)
plt.show()

print("\nPlot saved to per_cluster_radius_comparison.png")
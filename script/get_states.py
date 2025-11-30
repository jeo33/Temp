import numpy as np
import os
from read_data import *
import matplotlib.pyplot as plt

print(os.getcwd())
dir = f'..\\data\\sift\\1000_20_overlapping\\partitions'
groud_dir = f'..\\data\\sift\\sift_groundtruth.ivecs'
base = f'..\\data\\sift\\sift_base.fvecs'
query_path = f'..\\data\\sift\\sift_query.fvecs'
stats_dir = f'..\\data\\sift\\1000_20_overlapping\\cluster_stats'

os.makedirs(stats_dir, exist_ok=True)

g = inspect_file_data(groud_dir)
b = inspect_file_data(base)
q = inspect_file_data(query_path)

# Load all partitions and compute/save statistics
centroids = []
all_ids = []
cov_matrices = []
inv_cov_matrices = []
effective_radii = []

print("Computing cluster statistics...")
for it in range(1000):
    vecs = os.path.join(dir, f'partition_{it}.fvecs')
    ids = os.path.join(dir, f'partition_{it}.ids')
    v = inspect_file_data(vecs)
    i = inspect_file_data(ids)

    # Centroid
    centroid = np.mean(v, axis=0)
    centroids.append(centroid)
    all_ids.append(set(i))

    # Covariance matrix
    cov_matrix = np.cov(v, rowvar=False)
    cov_matrices.append(cov_matrix)

    # Regularized inverse covariance (add small value for numerical stability)
    reg_cov = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6
    inv_cov = np.linalg.inv(reg_cov)
    inv_cov_matrices.append(inv_cov)

    # Effective radius: mean distance from centroid to all points in cluster
    distances_to_centroid = np.linalg.norm(v - centroid, axis=1)
    effective_radius = np.mean(distances_to_centroid)
    effective_radii.append(effective_radius)

    # Save statistics
    np.save(os.path.join(stats_dir, f'centroid_{it}.npy'), centroid)
    np.save(os.path.join(stats_dir, f'cov_{it}.npy'), cov_matrix)
    np.save(os.path.join(stats_dir, f'inv_cov_{it}.npy'), inv_cov)
    np.save(os.path.join(stats_dir, f'effective_radius_{it}.npy'), effective_radius)

    if it % 20 == 0:
        print(f"  Partition {it}: radius={effective_radius:.2f}")

centroids = np.array(centroids)
effective_radii = np.array(effective_radii)

# Save global stats
np.save(os.path.join(stats_dir, 'all_centroids.npy'), centroids)
np.save(os.path.join(stats_dir, 'all_effective_radii.npy'), effective_radii)

print(f"\nStatistics saved to {stats_dir}")
print(f"Mean effective radius: {np.mean(effective_radii):.2f}")

# Test with 100 queries
num_test_queries = 100
k_clusters = 10
k_neighbors = 100


def euclidean_distance_to_centroid(query, centroids):
    """Simple L2 distance to centroid"""
    return np.linalg.norm(centroids - query, axis=1)


def mahalanobis_distance(query, centroids, inv_cov_matrices):
    """Mahalanobis distance accounting for cluster shape"""
    distances = []
    for i in range(len(centroids)):
        diff = query - centroids[i]
        dist = np.sqrt(np.dot(np.dot(diff, inv_cov_matrices[i]), diff))
        distances.append(dist)
    return np.array(distances)


def gap_based_distance(query, centroids, effective_radii):
    """Distance to cluster boundary: centroid_dist - radius"""
    centroid_dists = np.linalg.norm(centroids - query, axis=1)
    # Gap = distance to centroid - effective radius (negative means inside cluster)
    gaps = centroid_dists - effective_radii
    return gaps


def combined_distance(query, centroids, inv_cov_matrices, effective_radii, alpha=0.5):
    """Combine Mahalanobis with gap-based approach"""
    mahal_dists = mahalanobis_distance(query, centroids, inv_cov_matrices)
    gaps = gap_based_distance(query, centroids, effective_radii)
    # Normalize both
    mahal_norm = (mahal_dists - np.mean(mahal_dists)) / (np.std(mahal_dists) + 1e-10)
    gap_norm = (gaps - np.mean(gaps)) / (np.std(gaps) + 1e-10)
    return alpha * mahal_norm + (1 - alpha) * gap_norm


# Run comparison
results = {
    'euclidean': {'overlaps': [], 'recalls': []},
    'mahalanobis': {'overlaps': [], 'recalls': []},
    'gap_based': {'overlaps': [], 'recalls': []},
    'combined': {'overlaps': [], 'recalls': []},
    'oracle': {'recalls': []}
}

print(f"\nTesting {num_test_queries} queries...")
for qid in range(num_test_queries):
    query = q[qid]
    gt_neighbors = set(g[qid])

    # Oracle: actual best clusters
    gt_counts = np.array([len(gt_neighbors & all_ids[cid]) for cid in range(100)])
    best_actual_clusters = set(np.argsort(gt_counts)[::-1][:k_clusters])

    ids_in_oracle = set()
    for cid in best_actual_clusters:
        ids_in_oracle |= all_ids[cid]
    oracle_recall = len(gt_neighbors & ids_in_oracle) / k_neighbors
    results['oracle']['recalls'].append(oracle_recall)

    # Method 1: Euclidean distance to centroid
    euc_dists = euclidean_distance_to_centroid(query, centroids)
    euc_clusters = set(np.argsort(euc_dists)[:k_clusters])
    euc_overlap = len(euc_clusters & best_actual_clusters)
    ids_in_euc = set()
    for cid in euc_clusters:
        ids_in_euc |= all_ids[cid]
    euc_recall = len(gt_neighbors & ids_in_euc) / k_neighbors
    results['euclidean']['overlaps'].append(euc_overlap)
    results['euclidean']['recalls'].append(euc_recall)

    # Method 2: Mahalanobis distance
    mahal_dists = mahalanobis_distance(query, centroids, inv_cov_matrices)
    mahal_clusters = set(np.argsort(mahal_dists)[:k_clusters])
    mahal_overlap = len(mahal_clusters & best_actual_clusters)
    ids_in_mahal = set()
    for cid in mahal_clusters:
        ids_in_mahal |= all_ids[cid]
    mahal_recall = len(gt_neighbors & ids_in_mahal) / k_neighbors
    results['mahalanobis']['overlaps'].append(mahal_overlap)
    results['mahalanobis']['recalls'].append(mahal_recall)

    # Method 3: Gap-based (distance to boundary)
    gap_dists = gap_based_distance(query, centroids, effective_radii)
    gap_clusters = set(np.argsort(gap_dists)[:k_clusters])
    gap_overlap = len(gap_clusters & best_actual_clusters)
    ids_in_gap = set()
    for cid in gap_clusters:
        ids_in_gap |= all_ids[cid]
    gap_recall = len(gt_neighbors & ids_in_gap) / k_neighbors
    results['gap_based']['overlaps'].append(gap_overlap)
    results['gap_based']['recalls'].append(gap_recall)

    # Method 4: Combined
    comb_dists = combined_distance(query, centroids, inv_cov_matrices, effective_radii)
    comb_clusters = set(np.argsort(comb_dists)[:k_clusters])
    comb_overlap = len(comb_clusters & best_actual_clusters)
    ids_in_comb = set()
    for cid in comb_clusters:
        ids_in_comb |= all_ids[cid]
    comb_recall = len(gt_neighbors & ids_in_comb) / k_neighbors
    results['combined']['overlaps'].append(comb_overlap)
    results['combined']['recalls'].append(comb_recall)

# Print summary
print("\n" + "=" * 70)
print(f"{'Method':<15} {'Overlap (avg)':<15} {'Recall (avg)':<15} {'Gap to Oracle':<15}")
print("=" * 70)

oracle_mean = np.mean(results['oracle']['recalls']) * 100

for method in ['euclidean', 'mahalanobis', 'gap_based', 'combined']:
    overlap_mean = np.mean(results[method]['overlaps'])
    recall_mean = np.mean(results[method]['recalls']) * 100
    gap = oracle_mean - recall_mean
    print(f"{method:<15} {overlap_mean:<15.2f} {recall_mean:<14.1f}% {gap:<14.1f}%")

print(f"{'oracle':<15} {10:<15} {oracle_mean:<14.1f}% {0:<14.1f}%")
print("=" * 70)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

methods = ['euclidean', 'mahalanobis', 'gap_based', 'combined']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Plot 1: Overlap comparison
ax1 = axes[0, 0]
overlap_data = [results[m]['overlaps'] for m in methods]
bp1 = ax1.boxplot(overlap_data, labels=['Euclidean', 'Mahalanobis', 'Gap-based', 'Combined'], patch_artist=True)
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.set_ylabel('Cluster Overlap with Oracle')
ax1.set_title('Cluster Selection Overlap (k=10)')
ax1.axhline(10, color='red', linestyle='--', alpha=0.5, label='Perfect')
ax1.legend()

# Plot 2: Recall comparison
ax2 = axes[0, 1]
recall_data = [np.array(results[m]['recalls']) * 100 for m in methods]
recall_data.append(np.array(results['oracle']['recalls']) * 100)
bp2 = ax2.boxplot(recall_data, labels=['Euclidean', 'Mahalanobis', 'Gap-based', 'Combined', 'Oracle'],
                  patch_artist=True)
for patch, color in zip(bp2['boxes'], colors + ['#9467bd']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_ylabel('Recall@100 (%)')
ax2.set_title('Recall Distribution by Method')

# Plot 3: Per-query recall comparison
ax3 = axes[1, 0]
x = np.arange(num_test_queries)
ax3.plot(x, np.array(results['oracle']['recalls']) * 100, 'k-', alpha=0.3, label='Oracle', linewidth=2)
ax3.plot(x, np.array(results['euclidean']['recalls']) * 100, '-', color=colors[0], alpha=0.7, label='Euclidean')
ax3.plot(x, np.array(results['mahalanobis']['recalls']) * 100, '-', color=colors[1], alpha=0.7, label='Mahalanobis')
ax3.plot(x, np.array(results['gap_based']['recalls']) * 100, '-', color=colors[2], alpha=0.7, label='Gap-based')
ax3.set_xlabel('Query ID')
ax3.set_ylabel('Recall@100 (%)')
ax3.set_title('Per-Query Recall')
ax3.legend()

# Plot 4: Improvement over Euclidean
ax4 = axes[1, 1]
euc_recalls = np.array(results['euclidean']['recalls'])
improvements = {
    'Mahalanobis': np.array(results['mahalanobis']['recalls']) - euc_recalls,
    'Gap-based': np.array(results['gap_based']['recalls']) - euc_recalls,
    'Combined': np.array(results['combined']['recalls']) - euc_recalls
}
improvement_data = [improvements[m] * 100 for m in ['Mahalanobis', 'Gap-based', 'Combined']]
bp4 = ax4.boxplot(improvement_data, labels=['Mahalanobis', 'Gap-based', 'Combined'], patch_artist=True)
for patch, color in zip(bp4['boxes'], colors[1:]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
ax4.set_ylabel('Recall Improvement over Euclidean (%)')
ax4.set_title('Improvement vs Euclidean Baseline')

plt.tight_layout()
plt.savefig('covariance_comparison.png', dpi=150)
plt.show()

print("\nPlot saved to covariance_comparison.png")
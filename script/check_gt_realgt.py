import numpy as np
import os
from read_data import *
import matplotlib.pyplot as plt

print(os.getcwd())
dir = f'..\\data\\sift\\1000_20\\partitions'
groud_dir = f'..\\data\\sift\\sift_groundtruth.ivecs'
base = f'..\\data\\sift\\sift_base.fvecs'
query_path = f'..\\data\\sift\\sift_query.fvecs'

g = inspect_file_data(groud_dir)  # shape: (num_queries, 100)
b = inspect_file_data(base)
q = inspect_file_data(query_path)  # shape: (num_queries, dim)

# Configuration for 1000 partitions
num_partitions = 1000

# Load all partition ids and compute centroids
centroids = []
all_ids = []

print(f"Loading {num_partitions} partitions...")
for it in range(num_partitions):
    vecs_path = os.path.join(dir, f'partition_{it}.fvecs')
    ids_path = os.path.join(dir, f'partition_{it}.ids')
    v = inspect_file_data(vecs_path)
    i = inspect_file_data(ids_path)

    centroid = np.mean(v, axis=0)
    centroids.append(centroid)
    all_ids.append(set(i))  # use set for fast lookup

    if (it + 1) % 100 == 0:
        print(f"  Loaded {it + 1}/{num_partitions} partitions")

centroids = np.array(centroids)  # shape: (1000, dim)
print(f"Loaded {len(centroids)} centroids")

# For each query, compare nearest centroids vs actual best clusters
num_queries = len(q)
k_neighbors = 10  # ground truth neighbors to find

# Test multiple k_clusters values
k_clusters_list = [10, 20, 30, 50, 100]

results = {}

for k_clusters in k_clusters_list:
    overlaps = []
    recall_by_centroid = []
    recall_by_actual = []

    print(f"\nTesting k_clusters = {k_clusters}...")

    for qid in range(num_queries):
        query = q[qid]
        gt_neighbors = set(g[qid][:k_neighbors])  # top k ground truth neighbor ids

        # Method 1: Top k clusters by centroid distance
        distances_to_centroids = np.linalg.norm(centroids - query, axis=1)
        nearest_centroid_clusters = np.argsort(distances_to_centroids)[:k_clusters]

        # Method 2: Top k clusters by actual ground truth count
        gt_counts = []
        for cid in range(num_partitions):
            count = len(gt_neighbors & all_ids[cid])
            gt_counts.append(count)
        gt_counts = np.array(gt_counts)
        best_actual_clusters = np.argsort(gt_counts)[::-1][:k_clusters]

        # Calculate overlap between the two methods
        overlap = len(set(nearest_centroid_clusters) & set(best_actual_clusters))
        overlaps.append(overlap)

        # Calculate recall: how many GT neighbors are in selected clusters
        # Recall using centroid method
        ids_in_centroid_clusters = set()
        for cid in nearest_centroid_clusters:
            ids_in_centroid_clusters |= all_ids[cid]
        recall_centroid = len(gt_neighbors & ids_in_centroid_clusters) / k_neighbors
        recall_by_centroid.append(recall_centroid)

        # Recall using actual best clusters (oracle)
        ids_in_actual_clusters = set()
        for cid in best_actual_clusters:
            ids_in_actual_clusters |= all_ids[cid]
        recall_actual = len(gt_neighbors & ids_in_actual_clusters) / k_neighbors
        recall_by_actual.append(recall_actual)

    results[k_clusters] = {
        'overlaps': np.array(overlaps),
        'recall_centroid': np.array(recall_by_centroid),
        'recall_oracle': np.array(recall_by_actual)
    }

# Print summary
print("\n" + "=" * 90)
print(f"{'k_clusters':<12} {'Overlap':<15} {'Centroid Recall':<20} {'Oracle Recall':<20} {'Gap':<10}")
print("=" * 90)

for k_clusters in k_clusters_list:
    r = results[k_clusters]
    overlap_mean = np.mean(r['overlaps'])
    centroid_mean = np.mean(r['recall_centroid']) * 100
    oracle_mean = np.mean(r['recall_oracle']) * 100
    gap = oracle_mean - centroid_mean
    print(f"{k_clusters:<12} {overlap_mean:<15.2f} {centroid_mean:<19.1f}% {oracle_mean:<19.1f}% {gap:<9.1f}%")

print("=" * 90)

# Detailed summary for default k
k_clusters = 20
r = results[k_clusters]
print(f"\n=== Detailed Summary for k_clusters={k_clusters} ===")
print(f"Cluster Overlap (out of {k_clusters}):")
print(f"  Mean: {np.mean(r['overlaps']):.2f}, Min: {np.min(r['overlaps'])}, Max: {np.max(r['overlaps'])}")
print(f"\nRecall@{k_neighbors} using top {k_clusters} clusters:")
print(
    f"  Centroid method: Mean={np.mean(r['recall_centroid']) * 100:.1f}%, Min={np.min(r['recall_centroid']) * 100:.1f}%, Max={np.max(r['recall_centroid']) * 100:.1f}%")
print(
    f"  Oracle (actual):  Mean={np.mean(r['recall_oracle']) * 100:.1f}%, Min={np.min(r['recall_oracle']) * 100:.1f}%, Max={np.max(r['recall_oracle']) * 100:.1f}%")

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Histogram of cluster overlap for k=20
ax1 = axes[0, 0]
r = results[20]
ax1.hist(r['overlaps'], bins=range(0, 22), edgecolor='black', alpha=0.7)
ax1.set_xlabel('Number of Overlapping Clusters')
ax1.set_ylabel('Number of Queries')
ax1.set_title(f'Overlap: Nearest vs Best (k=20, {num_partitions} partitions)')
ax1.axvline(np.mean(r['overlaps']), color='red', linestyle='--', label=f'Mean={np.mean(r["overlaps"]):.2f}')
ax1.legend()

# Plot 2: Recall comparison histogram for k=20
ax2 = axes[0, 1]
ax2.hist(r['recall_centroid'] * 100, bins=20, alpha=0.6, label='Centroid Method', edgecolor='black')
ax2.hist(r['recall_oracle'] * 100, bins=20, alpha=0.6, label='Oracle (Actual Best)', edgecolor='black')
ax2.set_xlabel('Recall (%)')
ax2.set_ylabel('Number of Queries')
ax2.set_title(f'Recall@{k_neighbors} Distribution (k=20)')
ax2.legend()

# Plot 3: Scatter plot - Centroid recall vs Oracle recall
ax3 = axes[0, 2]
ax3.scatter(r['recall_oracle'] * 100, r['recall_centroid'] * 100, alpha=0.5, s=10)
ax3.plot([0, 100], [0, 100], 'r--', label='Perfect')
ax3.set_xlabel('Oracle Recall (%)')
ax3.set_ylabel('Centroid Recall (%)')
ax3.set_title('Centroid vs Oracle Recall per Query (k=20)')
ax3.legend()
ax3.set_xlim(0, 105)
ax3.set_ylim(0, 105)

# Plot 4: Recall vs k_clusters
ax4 = axes[1, 0]
centroid_recalls = [np.mean(results[k]['recall_centroid']) * 100 for k in k_clusters_list]
oracle_recalls = [np.mean(results[k]['recall_oracle']) * 100 for k in k_clusters_list]
ax4.plot(k_clusters_list, centroid_recalls, 'o-', label='Centroid Method', linewidth=2, markersize=8)
ax4.plot(k_clusters_list, oracle_recalls, 's--', label='Oracle', linewidth=2, markersize=8)
ax4.set_xlabel('Number of Clusters (k)')
ax4.set_ylabel(f'Recall@{k_neighbors} (%)')
ax4.set_title('Recall vs Number of Clusters')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Gap vs k_clusters
ax5 = axes[1, 1]
gaps = [np.mean(results[k]['recall_oracle'] - results[k]['recall_centroid']) * 100 for k in k_clusters_list]
ax5.bar(range(len(k_clusters_list)), gaps, tick_label=[str(k) for k in k_clusters_list], alpha=0.7, edgecolor='black')
ax5.set_xlabel('Number of Clusters (k)')
ax5.set_ylabel('Recall Gap (Oracle - Centroid) %')
ax5.set_title('Recall Gap vs k')
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Overlap vs k_clusters
ax6 = axes[1, 2]
overlap_means = [np.mean(results[k]['overlaps']) for k in k_clusters_list]
overlap_pcts = [np.mean(results[k]['overlaps']) / k * 100 for k in k_clusters_list]
ax6.bar(range(len(k_clusters_list)), overlap_pcts, tick_label=[str(k) for k in k_clusters_list], alpha=0.7,
        edgecolor='black', color='green')
ax6.set_xlabel('Number of Clusters (k)')
ax6.set_ylabel('Overlap Percentage (%)')
ax6.set_title('Cluster Selection Overlap (as % of k)')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('cluster_analysis_1000.png', dpi=150)
plt.show()

print("\nPlot saved to cluster_analysis_1000.png")

# Additional analysis: vectors visited
print("\n=== Vectors Visited Analysis ===")
cluster_sizes = [len(ids) for ids in all_ids]
print(
    f"Cluster sizes - Mean: {np.mean(cluster_sizes):.0f}, Min: {np.min(cluster_sizes)}, Max: {np.max(cluster_sizes)}, Std: {np.std(cluster_sizes):.0f}")

for k_clusters in k_clusters_list:
    avg_vectors = k_clusters * np.mean(cluster_sizes)
    print(
        f"k={k_clusters}: ~{avg_vectors:.0f} vectors visited on average ({avg_vectors / len(b) * 100:.2f}% of dataset)")
import numpy as np
import os
from read_data import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA


def euclidean_to_centroid(query, centroids):
    return np.linalg.norm(centroids - query, axis=1)


print(os.getcwd())

dir = f'..\\data\\sift\\1000_20\\partitions'
groud_dir = f'..\\data\\sift\\sift_groundtruth.ivecs'
query_path = f'..\\data\\sift\\sift_query.fvecs'
base_path = f'..\\data\\sift\\sift_base.fvecs'
stats_dir = f'..\\data\\sift\\1000_20\\cluster_stats'

g = inspect_file_data(groud_dir)
q = inspect_file_data(query_path)
base = inspect_file_data(base_path)

# Load cluster data
centroids = []
all_ids = []
all_vecs = []
principal_components = []
eigenvalues_list = []

for it in range(1000):
    ids = inspect_file_data(os.path.join(dir, f'partition_{it}.ids'))
    vecs = inspect_file_data(os.path.join(dir, f'partition_{it}.fvecs'))
    all_ids.append(ids)
    all_vecs.append(vecs)

    centroids.append(np.load(os.path.join(stats_dir, f'centroid_{it}.npy')))
    principal_components.append(np.load(os.path.join(stats_dir, f'pca_{it}.npy')))
    eigenvalues_list.append(np.load(os.path.join(stats_dir, f'eigenvalues_{it}.npy')))

centroids = np.array(centroids)

# Build reverse index: vector_id -> cluster_id
id_to_cluster = {}
for cid in range(1000):
    for vid in all_ids[cid]:
        id_to_cluster[vid] = cid

# Visualize first 10 queries
for qid in range(10):
    query = q[qid]
    gt_top10_ids = g[qid][:10]

    # Get distances and rankings
    euc_dists = euclidean_to_centroid(query, centroids)
    euc_ranking = np.argsort(euc_dists)
    top10_clusters = set(euc_ranking[:10])

    # Find GT clusters
    gt_clusters = set()
    gt_vectors = []
    for gt_id in gt_top10_ids:
        cid = id_to_cluster.get(gt_id)
        if cid is not None:
            gt_clusters.add(cid)
        gt_vectors.append(base[gt_id])
    gt_vectors = np.array(gt_vectors)

    # Clusters to visualize: top 10 by euclidean + GT clusters
    clusters_to_viz = list(top10_clusters | gt_clusters)

    # Collect all relevant vectors for PCA
    all_points = [query]  # query is first
    all_points.extend(gt_vectors)  # GT vectors

    cluster_start_idx = {}
    for cid in clusters_to_viz:
        cluster_start_idx[cid] = len(all_points)
        all_points.extend(all_vecs[cid])

    all_points.append(
        centroids[clusters_to_viz].reshape(-1, centroids.shape[1]) if len(clusters_to_viz) == 1 else np.vstack(
            [centroids[cid] for cid in clusters_to_viz]))

    # Rebuild with centroids
    all_points = [query]
    all_points.extend(gt_vectors)
    for cid in clusters_to_viz:
        all_points.append(centroids[cid])

    # Sample points from each cluster for visualization
    cluster_samples = {}
    for cid in clusters_to_viz:
        vecs = all_vecs[cid]
        if len(vecs) > 500:
            idx = np.random.choice(len(vecs), 500, replace=False)
            cluster_samples[cid] = vecs[idx]
        else:
            cluster_samples[cid] = vecs
        all_points.extend(cluster_samples[cid])

    all_points = np.array(all_points)

    # Fit PCA on all points
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_points)

    # Extract 2D coordinates
    query_2d = all_2d[0]
    gt_2d = all_2d[1:1 + len(gt_vectors)]

    idx = 1 + len(gt_vectors)
    centroids_2d = {}
    for cid in clusters_to_viz:
        centroids_2d[cid] = all_2d[idx]
        idx += 1

    cluster_points_2d = {}
    for cid in clusters_to_viz:
        n_samples = len(cluster_samples[cid])
        cluster_points_2d[cid] = all_2d[idx:idx + n_samples]
        idx += n_samples

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Color scheme
    colors = plt.cm.tab20(np.linspace(0, 1, len(clusters_to_viz)))
    cluster_colors = {cid: colors[i] for i, cid in enumerate(clusters_to_viz)}

    # Plot each cluster
    for cid in clusters_to_viz:
        points = cluster_points_2d[cid]
        centroid = centroids_2d[cid]

        in_top10 = cid in top10_clusters
        has_gt = cid in gt_clusters

        # Determine style
        if in_top10 and has_gt:
            edge_color = 'green'
            label_prefix = "✓ HIT"
            alpha = 0.6
        elif in_top10 and not has_gt:
            edge_color = 'red'
            label_prefix = "✗ FAKE"
            alpha = 0.4
        elif not in_top10 and has_gt:
            edge_color = 'orange'
            label_prefix = "! MISSED"
            alpha = 0.6
        else:
            edge_color = 'gray'
            label_prefix = ""
            alpha = 0.3

        # Plot cluster points
        ax.scatter(points[:, 0], points[:, 1], s=5, alpha=alpha,
                   color=cluster_colors[cid], label=None)

        # Fit ellipse to cluster points (using covariance)
        if len(points) > 2:
            cov = np.cov(points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 4 * np.sqrt(eigenvalues)  # 2 std = ~95%

            ellipse = Ellipse(xy=centroid, width=width, height=height, angle=angle,
                              fill=False, edgecolor=edge_color, linewidth=2, linestyle='-')
            ax.add_patch(ellipse)

        # Plot centroid
        ax.scatter(centroid[0], centroid[1], s=150, marker='X',
                   color=cluster_colors[cid], edgecolor='black', linewidth=1, zorder=10)

        # Label
        euc_rank = np.sum(euc_dists < euc_dists[cid]) + 1
        gt_count = sum(1 for gt_id in gt_top10_ids if id_to_cluster.get(gt_id) == cid)
        label = f"C{cid} (rank={euc_rank}, GT={gt_count}) {label_prefix}"
        ax.annotate(label, centroid, fontsize=8, ha='center', va='bottom',
                    xytext=(0, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # Draw line from query to centroid
        ax.plot([query_2d[0], centroid[0]], [query_2d[1], centroid[1]],
                color=edge_color, linestyle='--', alpha=0.5, linewidth=1)

    # Plot query
    ax.scatter(query_2d[0], query_2d[1], s=300, marker='*', color='blue',
               edgecolor='black', linewidth=2, zorder=20, label='Query')
    ax.annotate('QUERY', query_2d, fontsize=12, fontweight='bold', ha='center', va='bottom',
                xytext=(0, 15), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.9))

    # Plot GT vectors
    ax.scatter(gt_2d[:, 0], gt_2d[:, 1], s=100, marker='D', color='lime',
               edgecolor='black', linewidth=1, zorder=15, label='GT Neighbors')

    # Add GT rank labels
    for i, (x, y) in enumerate(gt_2d):
        ax.annotate(f'{i + 1}', (x, y), fontsize=7, ha='center', va='center', fontweight='bold')

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', markersize=15, label='Query'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='lime', markersize=10, label='GT Neighbors'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='gray', markersize=10, label='Centroid'),
        Line2D([0], [0], color='green', linewidth=2, label='HIT cluster (top10 + has GT)'),
        Line2D([0], [0], color='red', linewidth=2, label='FAKE cluster (top10 + no GT)'),
        Line2D([0], [0], color='orange', linewidth=2, label='MISSED cluster (not top10 + has GT)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Title and labels
    found_gt = sum(1 for cid in gt_clusters if cid in top10_clusters)
    total_gt_found = sum(1 for gt_id in gt_top10_ids if id_to_cluster.get(gt_id) in top10_clusters)
    ax.set_title(f'Query {qid} - Recall@10: {total_gt_found}/10 | '
                 f'GT spread across {len(gt_clusters)} clusters | '
                 f'{len(top10_clusters - gt_clusters)} fake in top-10 | '
                 f'{len(gt_clusters - top10_clusters)} missed', fontsize=12)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'query_{qid}_cluster_viz.png', dpi=150)
    plt.close()

    print(f"Query {qid}: Saved to query_{qid}_cluster_viz.png")

print("\nDone! Generated visualizations for queries 0-9")

# Also create a combined view for one query with more detail
qid = 0  # Pick query with interesting patterns
query = q[qid]
gt_top10_ids = g[qid][:10]

euc_dists = euclidean_to_centroid(query, centroids)
euc_ranking = np.argsort(euc_dists)
top10_clusters = set(euc_ranking[:10])

gt_clusters = set()
for gt_id in gt_top10_ids:
    cid = id_to_cluster.get(gt_id)
    if cid is not None:
        gt_clusters.add(cid)

clusters_to_viz = list(top10_clusters | gt_clusters)

# Create detailed 2x2 view
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Collect all points
all_points = [query]
gt_vectors = np.array([base[gt_id] for gt_id in gt_top10_ids])
all_points.extend(gt_vectors)

for cid in clusters_to_viz:
    all_points.append(centroids[cid])
    vecs = all_vecs[cid]
    if len(vecs) > 300:
        idx = np.random.choice(len(vecs), 300, replace=False)
        all_points.extend(vecs[idx])
    else:
        all_points.extend(vecs)

all_points = np.array(all_points)
pca = PCA(n_components=2)
all_2d = pca.fit_transform(all_points)

query_2d = all_2d[0]
gt_2d = all_2d[1:11]

# Panel 1: Full view with all clusters
ax1 = axes[0, 0]
idx = 11
for i, cid in enumerate(clusters_to_viz):
    centroid_2d = all_2d[idx]
    idx += 1
    n = min(300, len(all_vecs[cid]))
    points_2d = all_2d[idx:idx + n]
    idx += n

    in_top10 = cid in top10_clusters
    has_gt = cid in gt_clusters

    if in_top10 and has_gt:
        color = 'green'
    elif in_top10:
        color = 'red'
    else:
        color = 'orange'

    ax1.scatter(points_2d[:, 0], points_2d[:, 1], s=3, alpha=0.4, c=color)
    ax1.scatter(centroid_2d[0], centroid_2d[1], s=100, marker='X', c=color, edgecolor='black')
    ax1.annotate(f'C{cid}', centroid_2d, fontsize=8)

ax1.scatter(query_2d[0], query_2d[1], s=200, marker='*', c='blue', edgecolor='black', zorder=10)
ax1.scatter(gt_2d[:, 0], gt_2d[:, 1], s=50, marker='D', c='lime', edgecolor='black', zorder=10)
ax1.set_title(f'Query {qid}: Full View')
ax1.grid(True, alpha=0.3)

# Panel 2: Zoomed to query region
ax2 = axes[0, 1]
idx = 11
for i, cid in enumerate(clusters_to_viz):
    centroid_2d = all_2d[idx]
    idx += 1
    n = min(300, len(all_vecs[cid]))
    points_2d = all_2d[idx:idx + n]
    idx += n

    in_top10 = cid in top10_clusters
    has_gt = cid in gt_clusters

    if in_top10 and has_gt:
        color = 'green'
    elif in_top10:
        color = 'red'
    else:
        color = 'orange'

    ax2.scatter(points_2d[:, 0], points_2d[:, 1], s=5, alpha=0.5, c=color)
    ax2.scatter(centroid_2d[0], centroid_2d[1], s=100, marker='X', c=color, edgecolor='black')

ax2.scatter(query_2d[0], query_2d[1], s=200, marker='*', c='blue', edgecolor='black', zorder=10)
ax2.scatter(gt_2d[:, 0], gt_2d[:, 1], s=80, marker='D', c='lime', edgecolor='black', zorder=10)

# Zoom to query neighborhood
margin = 50
ax2.set_xlim(query_2d[0] - margin, query_2d[0] + margin)
ax2.set_ylim(query_2d[1] - margin, query_2d[1] + margin)
ax2.set_title(f'Query {qid}: Zoomed View')
ax2.grid(True, alpha=0.3)

# Panel 3: Distance bar chart
ax3 = axes[1, 0]
cluster_data = []
for cid in clusters_to_viz:
    in_top10 = cid in top10_clusters
    has_gt = cid in gt_clusters
    gt_count = sum(1 for gt_id in gt_top10_ids if id_to_cluster.get(gt_id) == cid)
    cluster_data.append({
        'cid': cid,
        'dist': euc_dists[cid],
        'in_top10': in_top10,
        'has_gt': has_gt,
        'gt_count': gt_count
    })

cluster_data.sort(key=lambda x: x['dist'])
x_pos = range(len(cluster_data))
colors = []
for d in cluster_data:
    if d['in_top10'] and d['has_gt']:
        colors.append('green')
    elif d['in_top10']:
        colors.append('red')
    else:
        colors.append('orange')

bars = ax3.bar(x_pos, [d['dist'] for d in cluster_data], color=colors, alpha=0.7)
ax3.set_xticks(x_pos)
ax3.set_xticklabels([f"C{d['cid']}\n({d['gt_count']})" for d in cluster_data], fontsize=8)
ax3.set_ylabel('Distance to Query')
ax3.set_title('Cluster Distances (label shows GT count)')
ax3.axhline(y=cluster_data[9]['dist'] if len(cluster_data) > 9 else 0, color='black', linestyle='--',
            label='Top-10 cutoff')

# Panel 4: Gap analysis
ax4 = axes[1, 1]
gap_data = []
for cid in clusters_to_viz:
    vecs = all_vecs[cid]
    centroid = centroids[cid]
    diff = query - centroid
    diff_norm = diff / (np.linalg.norm(diff) + 1e-10)

    vecs_centered = vecs - centroid
    projections = np.dot(vecs_centered, diff_norm)
    max_proj = np.max(projections)

    gap = euc_dists[cid] - max_proj

    in_top10 = cid in top10_clusters
    has_gt = cid in gt_clusters

    gap_data.append({
        'cid': cid,
        'dist': euc_dists[cid],
        'gap': gap,
        'in_top10': in_top10,
        'has_gt': has_gt
    })

for d in gap_data:
    if d['in_top10'] and d['has_gt']:
        color = 'green'
        marker = 'o'
    elif d['in_top10']:
        color = 'red'
        marker = 'x'
    else:
        color = 'orange'
        marker = '^'

    ax4.scatter(d['dist'], d['gap'], s=100, c=color, marker=marker)
    ax4.annotate(f"C{d['cid']}", (d['dist'], d['gap']), fontsize=8)

ax4.set_xlabel('Distance to Query')
ax4.set_ylabel('Gap (dist - boundary projection)')
ax4.set_title('Distance vs Gap')
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'query_{qid}_detailed_analysis.png', dpi=150)
plt.show()

print(f"\nSaved detailed analysis to query_{qid}_detailed_analysis.png")
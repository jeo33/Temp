import numpy as np
import os
from read_data import *
import matplotlib.pyplot as plt

def euclidean_to_centroid(query, centroids):
    return np.linalg.norm(centroids - query, axis=1)
print(os.getcwd())
dir = f'..\\data\\sift\\1000_20\\partitions'
groud_dir = f'..\\data\\sift\\sift_groundtruth.ivecs'
query_path = f'..\\data\\sift\\sift_query.fvecs'
stats_dir = f'..\\data\\sift\\1000_20\\cluster_stats'

g = inspect_file_data(groud_dir)
q = inspect_file_data(query_path)

# Load cluster data
centroids = []
all_ids = []
cluster_sizes = []
principal_components = []
eigenvalues_list = []
effective_radii = []

for it in range(100):
    ids = inspect_file_data(os.path.join(dir, f'partition_{it}.ids'))
    vecs = inspect_file_data(os.path.join(dir, f'partition_{it}.fvecs'))
    all_ids.append(set(ids))
    cluster_sizes.append(len(ids))

    centroid = np.load(os.path.join(stats_dir, f'centroid_{it}.npy'))
    centroids.append(centroid)
    principal_components.append(np.load(os.path.join(stats_dir, f'pca_{it}.npy')))
    eigenvalues_list.append(np.load(os.path.join(stats_dir, f'eigenvalues_{it}.npy')))

    # Compute various radius metrics
    dists_to_centroid = np.linalg.norm(vecs - centroid, axis=1)
    effective_radii.append({
        'mean': np.mean(dists_to_centroid),
        'std': np.std(dists_to_centroid),
        'min': np.min(dists_to_centroid),
        'max': np.max(dists_to_centroid),
        'median': np.median(dists_to_centroid),
        'p10': np.percentile(dists_to_centroid, 10),  # inner boundary
        'p90': np.percentile(dists_to_centroid, 90),  # outer boundary
    })

centroids = np.array(centroids)
cluster_sizes = np.array(cluster_sizes)

# Analyze FAKE vs REAL clusters
# Fake: ranked in top-10 by Euclidean but has 0 GT neighbors
# Real: ranked in top-10 and has GT neighbors

num_queries = len(q)
fake_clusters_data = []  # (query_id, cluster_id, euc_rank, euc_dist, features...)
real_clusters_data = []
missed_clusters_data = []  # has GT but not in top-10

print("Analyzing fake vs real clusters...")
for qid in range(num_queries):
    query = q[qid]
    gt_top10 = set(g[qid][:10])

    euc_dists = euclidean_to_centroid(query, centroids)
    euc_ranking = np.argsort(euc_dists)
    euc_top10 = set(euc_ranking[:10])

    gt_counts = np.array([len(gt_top10 & all_ids[cid]) for cid in range(100)])

    for rank, cid in enumerate(euc_ranking[:10]):
        dist = euc_dists[cid]
        gt_count = gt_counts[cid]

        # Compute features for this query-cluster pair
        diff = query - centroids[cid]
        diff_norm = diff / (np.linalg.norm(diff) + 1e-10)

        pc = principal_components[cid]
        eig = eigenvalues_list[cid]

        # Feature 1: variance ratio in query direction
        projections = np.dot(pc.T, diff_norm) ** 2
        var_in_query_dir = np.sum(projections * eig)
        total_var = np.sum(eig)
        var_ratio = var_in_query_dir / (total_var + 1e-10)

        # Feature 2: distance relative to cluster radius
        radius = effective_radii[cid]
        dist_to_mean_radius = dist - radius['mean']
        dist_to_inner = dist - radius['p10']
        dist_to_outer = dist - radius['p90']
        normalized_dist = dist / (radius['mean'] + 1e-10)

        # Feature 3: how much query is "inside" vs "outside" cluster
        # Project query onto principal components
        projected_dists = []
        for k in range(min(20, len(eig))):
            proj = np.abs(np.dot(diff, pc[:, k]))
            spread = np.sqrt(eig[k] + 1e-10)
            projected_dists.append(proj / spread)  # how many stds away
        avg_std_away = np.mean(projected_dists)
        max_std_away = np.max(projected_dists)

        # Feature 4: alignment with top principal components
        top_k_alignment = np.sum(projections[:5] * eig[:5]) / (np.sum(eig[:5]) + 1e-10)

        # Feature 5: cluster density (size / volume proxy)
        density_proxy = cluster_sizes[cid] / (radius['mean'] ** 2 + 1e-10)

        features = {
            'qid': qid,
            'cid': cid,
            'rank': rank + 1,
            'euc_dist': dist,
            'gt_count': gt_count,
            'var_ratio': var_ratio,
            'dist_to_mean_radius': dist_to_mean_radius,
            'dist_to_inner': dist_to_inner,
            'dist_to_outer': dist_to_outer,
            'normalized_dist': normalized_dist,
            'avg_std_away': avg_std_away,
            'max_std_away': max_std_away,
            'top_k_alignment': top_k_alignment,
            'density_proxy': density_proxy,
            'cluster_size': cluster_sizes[cid],
        }

        if gt_count == 0:
            fake_clusters_data.append(features)
        else:
            real_clusters_data.append(features)

    # Also collect missed clusters (has GT but not selected)
    for cid in range(100):
        if gt_counts[cid] > 0 and cid not in euc_top10:
            dist = euc_dists[cid]
            rank = np.sum(euc_dists < dist) + 1

            diff = query - centroids[cid]
            diff_norm = diff / (np.linalg.norm(diff) + 1e-10)
            pc = principal_components[cid]
            eig = eigenvalues_list[cid]

            projections = np.dot(pc.T, diff_norm) ** 2
            var_in_query_dir = np.sum(projections * eig)
            total_var = np.sum(eig)
            var_ratio = var_in_query_dir / (total_var + 1e-10)

            radius = effective_radii[cid]
            normalized_dist = dist / (radius['mean'] + 1e-10)

            projected_dists = []
            for k in range(min(20, len(eig))):
                proj = np.abs(np.dot(diff, pc[:, k]))
                spread = np.sqrt(eig[k] + 1e-10)
                projected_dists.append(proj / spread)
            avg_std_away = np.mean(projected_dists)

            missed_clusters_data.append({
                'qid': qid,
                'cid': cid,
                'rank': rank,
                'euc_dist': dist,
                'gt_count': gt_counts[cid],
                'var_ratio': var_ratio,
                'normalized_dist': normalized_dist,
                'avg_std_away': avg_std_away,
                'cluster_size': cluster_sizes[cid],
            })




print(f"\nTotal fake clusters (in top-10, 0 GT): {len(fake_clusters_data)}")
print(f"Total real clusters (in top-10, has GT): {len(real_clusters_data)}")
print(f"Total missed clusters (has GT, not in top-10): {len(missed_clusters_data)}")

# Statistical comparison
print("\n" + "=" * 80)
print("FEATURE COMPARISON: FAKE vs REAL CLUSTERS")
print("=" * 80)

features_to_compare = ['euc_dist', 'var_ratio', 'normalized_dist', 'avg_std_away',
                       'max_std_away', 'top_k_alignment', 'density_proxy', 'cluster_size',
                       'dist_to_mean_radius', 'dist_to_inner', 'dist_to_outer']

print(f"\n{'Feature':<22} {'Fake (mean)':<15} {'Real (mean)':<15} {'Diff':<10} {'Separable?':<12}")
print("-" * 80)

separable_features = []
for feat in features_to_compare:
    fake_vals = np.array([d[feat] for d in fake_clusters_data])
    real_vals = np.array([d[feat] for d in real_clusters_data])

    fake_mean = np.mean(fake_vals)
    real_mean = np.mean(real_vals)
    fake_std = np.std(fake_vals)
    real_std = np.std(real_vals)

    # Check if distributions are separable
    # Using Cohen's d effect size
    pooled_std = np.sqrt((fake_std ** 2 + real_std ** 2) / 2)
    cohens_d = abs(fake_mean - real_mean) / (pooled_std + 1e-10)

    separable = "YES" if cohens_d > 0.5 else "no"
    if cohens_d > 0.5:
        separable_features.append((feat, cohens_d, fake_mean, real_mean))

    print(
        f"{feat:<22} {fake_mean:<15.3f} {real_mean:<15.3f} {fake_mean - real_mean:<10.3f} {separable} (d={cohens_d:.2f})")

# Detailed look at best separating features
print("\n" + "=" * 80)
print("BEST SEPARATING FEATURES (Cohen's d > 0.5)")
print("=" * 80)

for feat, d, fake_m, real_m in sorted(separable_features, key=lambda x: -x[1]):
    fake_vals = np.array([d[feat] for d in fake_clusters_data])
    real_vals = np.array([d[feat] for d in real_clusters_data])

    print(f"\n{feat} (Cohen's d = {d:.2f}):")
    print(
        f"  Fake: mean={fake_m:.3f}, std={np.std(fake_vals):.3f}, range=[{np.min(fake_vals):.3f}, {np.max(fake_vals):.3f}]")
    print(
        f"  Real: mean={real_m:.3f}, std={np.std(real_vals):.3f}, range=[{np.min(real_vals):.3f}, {np.max(real_vals):.3f}]")

# Plotting
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

plot_features = ['euc_dist', 'var_ratio', 'normalized_dist', 'avg_std_away',
                 'dist_to_mean_radius', 'dist_to_outer', 'cluster_size', 'top_k_alignment', 'density_proxy']

for idx, feat in enumerate(plot_features):
    ax = axes[idx // 3, idx % 3]

    fake_vals = [d[feat] for d in fake_clusters_data]
    real_vals = [d[feat] for d in real_clusters_data]

    ax.hist(fake_vals, bins=30, alpha=0.6, label=f'Fake (n={len(fake_vals)})', color='red')
    ax.hist(real_vals, bins=30, alpha=0.6, label=f'Real (n={len(real_vals)})', color='green')
    ax.set_xlabel(feat)
    ax.set_ylabel('Count')
    ax.legend()
    ax.set_title(feat)

plt.tight_layout()
plt.savefig('fake_vs_real_clusters.png', dpi=150)
plt.show()

# Try a simple pruning rule based on findings
print("\n" + "=" * 80)
print("TESTING PRUNING RULES")
print("=" * 80)


def test_pruning_rule(rule_func, rule_name):
    """Test a pruning rule: returns (precision, recall, f1)"""
    tp = sum(1 for d in fake_clusters_data if rule_func(d))  # correctly identified fake
    fp = sum(1 for d in real_clusters_data if rule_func(d))  # wrongly pruned real
    fn = sum(1 for d in fake_clusters_data if not rule_func(d))  # missed fake
    tn = sum(1 for d in real_clusters_data if not rule_func(d))  # correctly kept real

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    print(f"{rule_name:<40} Prec={precision:.2f} Rec={recall:.2f} F1={f1:.2f} | Pruned: {tp + fp} fake, {fp} real")
    return precision, recall, f1


# Test various pruning rules
print("\nRule: Prune if condition is TRUE")
print("-" * 80)

test_pruning_rule(lambda d: d['normalized_dist'] > 1.5, "normalized_dist > 1.5")
test_pruning_rule(lambda d: d['normalized_dist'] > 1.3, "normalized_dist > 1.3")
test_pruning_rule(lambda d: d['normalized_dist'] > 1.2, "normalized_dist > 1.2")
test_pruning_rule(lambda d: d['dist_to_outer'] > 0, "dist_to_outer > 0 (outside 90th)")
test_pruning_rule(lambda d: d['dist_to_mean_radius'] > 50, "dist_to_mean_radius > 50")
test_pruning_rule(lambda d: d['dist_to_mean_radius'] > 100, "dist_to_mean_radius > 100")
test_pruning_rule(lambda d: d['avg_std_away'] > 3, "avg_std_away > 3")
test_pruning_rule(lambda d: d['avg_std_away'] > 2.5, "avg_std_away > 2.5")
test_pruning_rule(lambda d: d['var_ratio'] < 0.015, "var_ratio < 0.015")
test_pruning_rule(lambda d: d['var_ratio'] < 0.02, "var_ratio < 0.02")
test_pruning_rule(lambda d: d['cluster_size'] < 8000, "cluster_size < 8000")

# Combined rules
test_pruning_rule(lambda d: d['normalized_dist'] > 1.3 and d['var_ratio'] < 0.02,
                  "normalized_dist > 1.3 AND var_ratio < 0.02")
test_pruning_rule(lambda d: d['dist_to_outer'] > 0 and d['avg_std_away'] > 2.5,
                  "dist_to_outer > 0 AND avg_std_away > 2.5")
test_pruning_rule(lambda d: d['normalized_dist'] > 1.2 or d['avg_std_away'] > 3,
                  "normalized_dist > 1.2 OR avg_std_away > 3")

# Now test actual recall improvement with best pruning rule
print("\n" + "=" * 80)
print("TESTING RECALL IMPROVEMENT WITH PRUNING")
print("=" * 80)


def evaluate_with_pruning(prune_func, k_select=10, k_final=10):
    """Select top k_select by Euclidean, prune, take top k_final remaining"""
    recalls = []

    for qid in range(num_queries):
        query = q[qid]
        gt_top10 = set(g[qid][:10])

        euc_dists = euclidean_to_centroid(query, centroids)
        euc_ranking = np.argsort(euc_dists)

        # Get candidates and compute features
        candidates = []
        for cid in euc_ranking[:k_select]:
            diff = query - centroids[cid]
            diff_norm = diff / (np.linalg.norm(diff) + 1e-10)
            pc = principal_components[cid]
            eig = eigenvalues_list[cid]

            projections = np.dot(pc.T, diff_norm) ** 2
            var_ratio = np.sum(projections * eig) / (np.sum(eig) + 1e-10)

            radius = effective_radii[cid]
            normalized_dist = euc_dists[cid] / (radius['mean'] + 1e-10)
            dist_to_outer = euc_dists[cid] - radius['p90']

            projected_dists = []
            for k in range(min(20, len(eig))):
                proj = np.abs(np.dot(diff, pc[:, k]))
                spread = np.sqrt(eig[k] + 1e-10)
                projected_dists.append(proj / spread)
            avg_std_away = np.mean(projected_dists)

            features = {
                'cid': cid,
                'euc_dist': euc_dists[cid],
                'var_ratio': var_ratio,
                'normalized_dist': normalized_dist,
                'dist_to_outer': dist_to_outer,
                'avg_std_away': avg_std_away,
            }

            if not prune_func(features):
                candidates.append((cid, euc_dists[cid]))

        # Take top k_final after pruning
        candidates.sort(key=lambda x: x[1])
        selected = set(cid for cid, _ in candidates[:k_final])

        # If we pruned too many, add back from original ranking
        if len(selected) < k_final:
            for cid in euc_ranking:
                if cid not in selected:
                    selected.add(cid)
                if len(selected) >= k_final:
                    break

        # Calculate recall
        ids_in_selected = set()
        for cid in selected:
            ids_in_selected |= all_ids[cid]

        recall = len(gt_top10 & ids_in_selected) / 10
        recalls.append(recall)

    return np.mean(recalls) * 100


# Baseline
baseline = evaluate_with_pruning(lambda d: False, k_select=10, k_final=10)
print(f"Baseline (no pruning, k=10): {baseline:.1f}%")

# Test pruning with expanded candidate set
print("\nWith k_select=15, k_final=10:")
print(f"  No pruning: {evaluate_with_pruning(lambda d: False, k_select=15, k_final=10):.1f}%")
print(
    f"  Prune normalized_dist > 1.3: {evaluate_with_pruning(lambda d: d['normalized_dist'] > 1.3, k_select=15, k_final=10):.1f}%")
print(
    f"  Prune avg_std_away > 3: {evaluate_with_pruning(lambda d: d['avg_std_away'] > 3, k_select=15, k_final=10):.1f}%")
print(
    f"  Prune dist_to_outer > 0: {evaluate_with_pruning(lambda d: d['dist_to_outer'] > 0, k_select=15, k_final=10):.1f}%")

print("\nWith k_select=20, k_final=10:")
print(f"  No pruning: {evaluate_with_pruning(lambda d: False, k_select=20, k_final=10):.1f}%")
print(
    f"  Prune normalized_dist > 1.3: {evaluate_with_pruning(lambda d: d['normalized_dist'] > 1.3, k_select=20, k_final=10):.1f}%")
print(
    f"  Prune avg_std_away > 3: {evaluate_with_pruning(lambda d: d['avg_std_away'] > 3, k_select=20, k_final=10):.1f}%")
print(
    f"  Prune combined: {evaluate_with_pruning(lambda d: d['normalized_dist'] > 1.3 and d['avg_std_away'] > 2.5, k_select=20, k_final=10):.1f}%")
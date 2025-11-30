import numpy as np
import os
from read_data import *

# 1. SETUP & DATA LOADING
# ---------------------------------------------------------
dir_path = f'..\\data\\sift\\1000_20\\partitions'
ground_dir = f'..\\data\\sift\\sift_groundtruth.ivecs'
query_path = f'..\\data\\sift\\sift_query.fvecs'
stats_dir = f'..\\data\\sift\\1000_20\\cluster_stats'

g = inspect_file_data(ground_dir)
q = inspect_file_data(query_path)

print("Loading cluster stats...")
centroids = []
all_ids = []
principal_components = []
inverse_eigenvalues = []

# Load data
for it in range(1000):
    ids = inspect_file_data(os.path.join(dir_path, f'partition_{it}.ids'))
    all_ids.append(set(ids))

    centroids.append(np.load(os.path.join(stats_dir, f'centroid_{it}.npy')))

    # PCA & Boundary Prep
    pc = np.load(os.path.join(stats_dir, f'pca_{it}.npy'))
    eig = np.load(os.path.join(stats_dir, f'eigenvalues_{it}.npy'))

    # 3-Sigma Boundary
    sigma = np.sqrt(eig)
    boundary_scaling = 3.0 * sigma
    inv_sq_sigma = 1.0 / (boundary_scaling ** 2 + 1e-9)

    principal_components.append(pc)
    inverse_eigenvalues.append(inv_sq_sigma)

centroids = np.array(centroids)


# ---------------------------------------------------------
# 2. BOUNDARY LOGIC
# ---------------------------------------------------------
def get_boundary_distance(diff_vec, pc, inv_sq_sigma):
    dist = np.linalg.norm(diff_vec)
    if dist < 1e-9: return 99999.0
    unit_u = diff_vec / dist
    projections = np.dot(unit_u, pc.T)
    ellipsoid_term = np.sum((projections ** 2) * inv_sq_sigma[:len(projections)])
    r_dir = 1.0 / np.sqrt(ellipsoid_term + 1e-12)
    return r_dir


# ---------------------------------------------------------
# 3. RECALL@100 EVALUATION
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("TESTING RECALL@100 (Upper Bound)")
print("=" * 60)


def evaluate_recall_100(k_select=50, k_final=20, penalty_weight=2.0):
    """
    Evaluates Recall@100.
    k_select: How many candidates to consider for re-ranking.
    k_final:  How many clusters we actually 'visit'.
    """
    recalls = []
    recall_depth = 100  # We are looking for the top 100 neighbors

    for qid in range(len(q)):
        query = q[qid]

        # --- KEY CHANGE: Slice to 100 instead of 10 ---
        gt_top100 = set(g[qid][:recall_depth])

        # 1. Euclidean Pre-selection
        dists = np.linalg.norm(centroids - query, axis=1)
        candidates_idx = np.argsort(dists)[:k_select]

        rerank_candidates = []

        for cid in candidates_idx:
            dist = dists[cid]
            diff = query - centroids[cid]

            # 2. Boundary Logic
            r_dir = get_boundary_distance(diff, principal_components[cid], inverse_eigenvalues[cid])
            gap = dist - r_dir

            if gap <= 0:
                final_score = dist  # Inside
            else:
                final_score = dist + (gap * penalty_weight)  # Outside -> Penalize

            rerank_candidates.append((cid, final_score))

        # 3. Sort and Select
        rerank_candidates.sort(key=lambda x: x[1])
        selected = set(cid for cid, score in rerank_candidates[:k_final])

        # 4. Calculate Recall@100
        ids_in_selected = set()
        for cid in selected:
            ids_in_selected |= all_ids[cid]

        # --- KEY CHANGE: Divide by 100 ---
        recall = len(gt_top100 & ids_in_selected) / recall_depth
        recalls.append(recall)

    return np.mean(recalls) * 100


# ---------------------------------------------------------
# 4. RUN COMPARISON
# ---------------------------------------------------------

# Scenario: We scan 50 candidates and pick the best 20 clusters to visit.
K_SCAN = 50
K_VISIT = 20

# Baseline (Penalty = 0.0 is pure Euclidean)



base_score = evaluate_recall_100(k_select=K_SCAN, k_final=K_VISIT, penalty_weight=0.0)
print(f"Baseline (Euclidean, Visit {K_VISIT}): {base_score:.2f}%")

# Test Weights
score_w1 = evaluate_recall_100(k_select=K_SCAN, k_final=K_VISIT, penalty_weight=1.0)
print(f"Penalty Weight 1.0 (Visit {K_VISIT}):  {score_w1:.2f}%")

score_w2 = evaluate_recall_100(k_select=K_SCAN, k_final=K_VISIT, penalty_weight=2.0)
print(f"Penalty Weight 2.0 (Visit {K_VISIT}):  {score_w2:.2f}%")

best_score = max(score_w1, score_w2)
print(f"\nImprovement on R@100: {best_score - base_score:.2f}%")
import numpy as np
import argparse
import os
from read_data import *

print(os.getcwd())
dir = f'..\\data\\sift\\1000_20\\partitions'
groud_dir = f'..\\data\\sift\\sift_groundtruth.ivecs'
base = f'..\\data\\sift\\sift_base.fvecs'

g = inspect_file_data(groud_dir)
b = inspect_file_data(base)

cluster_stats = []

for it in range(1000):
    vecs = os.path.join(dir, f'partition_{it}.fvecs')
    ids = os.path.join(dir, f'partition_{it}.ids')
    v = inspect_file_data(vecs)
    i = inspect_file_data(ids)

    if not np.array_equal(b[i], v):
        print(f"Partition {it}: mismatch found")
    else:
        print(f"Partition {it}: OK")

    # Calculate statistics for this cluster
    mean = np.mean(v, axis=0)                    # shape: (dim,)
    variance = np.var(v, axis=0)                 # shape: (dim,)
    total_variance = np.sum(variance)            # scalar
    cov_matrix = np.cov(v, rowvar=False)         # shape: (dim, dim)

    cluster_stats.append({
        'partition': it,
        'num_vectors': len(v),
        'mean': mean,
        'variance': variance,
        'total_variance': total_variance,
        'cov_matrix': cov_matrix
    })

    print(f"  Vectors: {len(v)}, Total Variance: {total_variance:.4f}")

# Summary statistics across all clusters
all_variances = [s['total_variance'] for s in cluster_stats]
print("\n=== Summary ===")
print(f"Mean cluster variance: {np.mean(all_variances):.4f}")
print(f"Min cluster variance: {np.min(all_variances):.4f} (partition {np.argmin(all_variances)})")
print(f"Max cluster variance: {np.max(all_variances):.4f} (partition {np.argmax(all_variances)})")

print("good")
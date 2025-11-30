import numpy as np
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
        continue

    # Calculate statistics
    mean = np.mean(v, axis=0)
    cov_matrix = np.cov(v, rowvar=False)

    # Eigenvalues reveal the shape
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending order

    # Shape metrics
    max_eig = eigenvalues[0]
    min_eig = eigenvalues[-1]
    median_eig = np.median(eigenvalues)

    # Condition number: ratio of max to min eigenvalue
    # High = pancake/elongated, Low (~1) = sphere
    condition_number = max_eig / (min_eig + 1e-10)

    # Effective dimensionality: how many dimensions carry variance
    # Sum of eigenvalues / max eigenvalue
    total_var = np.sum(eigenvalues)
    effective_dim = total_var / max_eig

    # Participation ratio: another measure of dimensionality
    participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)

    # Top-k variance ratio: what % of variance is in top k dimensions
    top_10_ratio = np.sum(eigenvalues[:10]) / total_var
    top_50_ratio = np.sum(eigenvalues[:50]) / total_var

    # Classify shape
    if condition_number < 10:
        shape = "Sphere"
    elif condition_number < 100:
        shape = "Ellipsoid"
    elif condition_number < 1000:
        shape = "Elongated"
    else:
        shape = "Pancake"

    cluster_stats.append({
        'partition': it,
        'num_vectors': len(v),
        'condition_number': condition_number,
        'effective_dim': effective_dim,
        'participation_ratio': participation_ratio,
        'top_10_ratio': top_10_ratio,
        'top_50_ratio': top_50_ratio,
        'shape': shape,
        'eigenvalues': eigenvalues
    })

    print(
        f"Partition {it:3d}: {shape:10s} | Cond#: {condition_number:10.1f} | EffDim: {effective_dim:5.1f} | Top10: {top_10_ratio * 100:5.1f}% | Top50: {top_50_ratio * 100:5.1f}%")

# Summary
print("\n=== Shape Summary ===")
shapes = [s['shape'] for s in cluster_stats]
for shape_type in ['Sphere', 'Ellipsoid', 'Elongated', 'Pancake']:
    count = shapes.count(shape_type)
    print(f"{shape_type}: {count} clusters ({count}%)")

cond_numbers = [s['condition_number'] for s in cluster_stats]
eff_dims = [s['effective_dim'] for s in cluster_stats]
print(
    f"\nCondition Number - Mean: {np.mean(cond_numbers):.1f}, Min: {np.min(cond_numbers):.1f}, Max: {np.max(cond_numbers):.1f}")
print(f"Effective Dim    - Mean: {np.mean(eff_dims):.1f}, Min: {np.min(eff_dims):.1f}, Max: {np.max(eff_dims):.1f}")
import numpy as np
import os
import matplotlib.pyplot as plt
from read_data import *

# 1. SETUP PATHS
# ---------------------------------------------------------
datasets = {
    'Standard': {
        'partitions': f'..\\data\\sift\\1000_20\\partitions',
        'stats': f'..\\data\\sift\\1000_20\\cluster_stats'
    },
    'Overlapping': {
        'partitions': f'..\\data\\sift\\1000_20_overlapping\\partitions',
        'stats': f'..\\data\\sift\\1000_20_overlapping\\cluster_stats'
    }
}

ground_dir = f'..\\data\\sift\\sift_groundtruth.ivecs'
query_path = f'..\\data\\sift\\sift_query.fvecs'

# Load Global Data
print("Loading Global Data...")
g = inspect_file_data(ground_dir)
q = inspect_file_data(query_path)
num_queries = len(q)


# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------
def load_cluster_data(partition_dir, stats_dir):
    print(f"Loading data from {partition_dir}...")
    centroids = []
    all_ids = []

    for it in range(1000):
        ids_path = os.path.join(partition_dir, f'partition_{it}.ids')
        ids = inspect_file_data(ids_path)
        all_ids.append(set(ids))

        cent_path = os.path.join(stats_dir, f'centroid_{it}.npy')
        # Fallback if specific stats don't exist
        if not os.path.exists(cent_path):
            # Assume standard stats path if overlapping folder doesn't have them
            cent_path = cent_path.replace('1000_20_overlapping', '1000_20')

        centroids.append(np.load(cent_path))

    return np.array(centroids), all_ids


def evaluate_dataset(centroids, all_ids, n_probes_list):
    avg_recalls = []

    print(f"Evaluating for n_probes: {n_probes_list}")

    for n_probe in n_probes_list:
        print(f"  Processing n_probe={n_probe}...", end='\r')
        current_recalls = []

        for qid in range(num_queries):
            query = q[qid]
            gt_top100 = set(g[qid][:100])

            # Route
            dists = np.linalg.norm(centroids - query, axis=1)
            nearest_clusters = np.argsort(dists)[:n_probe]

            # Search
            ids_found = set()
            for cid in nearest_clusters:
                ids_found |= all_ids[cid]

            # Recall
            recall = len(gt_top100 & ids_found) / 100
            current_recalls.append(recall)

        avg_recalls.append(np.mean(current_recalls) * 100)

    print("Done.                               ")
    return avg_recalls


# ---------------------------------------------------------
# 3. MAIN EXECUTION
# ---------------------------------------------------------
n_probes_to_test = [1, 5, 10, 20]
results = {}

for name, paths in datasets.items():
    print(f"\nProcessing {name} Dataset...")

    # Handle stats path check
    if 'overlapping' in name and not os.path.exists(paths['stats']):
        print(f"Redirecting stats path for {name} to standard directory.")
        paths['stats'] = datasets['Standard']['stats']

    centroids, cluster_ids = load_cluster_data(paths['partitions'], paths['stats'])
    results[name] = evaluate_dataset(centroids, cluster_ids, n_probes_to_test)

# ---------------------------------------------------------
# 4. PLOTTING (BAR CHART)
# ---------------------------------------------------------
print("\nPlotting results...")
plt.figure(figsize=(10, 6))

x = np.arange(len(n_probes_to_test))
width = 0.35

recalls_std = results['Standard']
recalls_ovr = results['Overlapping']

# Create bars
plt.bar(x - width / 2, recalls_std, width, label='Standard', color='skyblue', edgecolor='black')
plt.bar(x + width / 2, recalls_ovr, width, label='Overlapping', color='salmon', edgecolor='black')

# Add values on top of bars
for i, v in enumerate(recalls_std):
    plt.text(i - width / 2, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
for i, v in enumerate(recalls_ovr):
    plt.text(i + width / 2, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

plt.xlabel('n_probe (Clusters Visited)')
plt.ylabel('Recall@100 (%)')
plt.title('Impact of Overlapping Partitions on Recall')
plt.xticks(x, n_probes_to_test)
plt.legend()
plt.ylim(0, 100)  # Recall is 0-100
plt.grid(axis='y', linestyle='--', alpha=0.5)


plt.tight_layout()
plt.savefig('comparison_bar_nprobe.png', dpi=150)
plt.show()

print("Done. Saved plot to 'comparison_bar_nprobe.png'.")
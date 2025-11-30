import numpy as np
import faiss
import os
import urllib.request
import tarfile
import argparse


def download_data(data_dir, dataset):
    if dataset == "sift":
        url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
        filename = "sift.tar.gz"
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        print(f"Downloading {url} to {filepath}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading: {e}")
            return

        print(f"Extracting {filepath}...")
        try:
            with tarfile.open(filepath, "r:gz") as tar:
                tar.extractall(path=data_dir)
            print("Extraction complete.")
        except Exception as e:
            print(f"Error extracting: {e}")
    else:
        print(f"Dataset {dataset} not supported.")


def fvecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view('float32')


def fvecs_write(fname, m):
    m = m.astype(np.float32)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    d = m.shape[1]
    n = m.shape[0]
    header = np.array([d], dtype=np.int32)
    with open(fname, 'wb') as f:
        for i in range(n):
            header.tofile(f)
            m[i].tofile(f)


def ids_write(fname, ids):
    ids = ids.astype(np.int64)
    ids.tofile(fname)


def save_results(output_dir, centroids, assignments, data):
    # Standard save function for non-overlapping partitions
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    centroids_path = os.path.join(output_dir, "centroids.fvecs")
    fvecs_write(centroids_path, centroids)

    assignments_path = os.path.join(output_dir, "assignments.npy")
    np.save(assignments_path, assignments)

    partitions_dir = os.path.join(output_dir, "partitions")
    if not os.path.exists(partitions_dir):
        os.makedirs(partitions_dir)

    ids = np.arange(data.shape[0], dtype=np.int64)
    k = centroids.shape[0]
    print(f"Saving {k} partitions to {partitions_dir}...")

    for i in range(k):
        mask = (assignments == i)
        partition_data = data[mask]
        partition_ids = ids[mask]

        fvecs_write(os.path.join(partitions_dir, f"partition_{i}.fvecs"), partition_data)
        ids_write(os.path.join(partitions_dir, f"partition_{i}.ids"), partition_ids)

    print("Partitions and IDs saved.")


def partition_data(data_dir, dataset, k, niter):
    if dataset == "sift":
        dataset_dir = os.path.join(data_dir, "sift")
        learn_path = os.path.join(dataset_dir, "sift_learn.fvecs")
        base_path = os.path.join(dataset_dir, "sift_base.fvecs")
    else:
        print(f"Dataset {dataset} not supported.")
        return

    if os.path.exists(learn_path) and os.path.exists(base_path):
        print(f"Reading training data from {learn_path}...")
        x_learn = fvecs_read(learn_path)

        print(f"Training K-means with k={k}...")
        d = x_learn.shape[1]
        kmeans = faiss.Kmeans(d, k, niter=niter, verbose=True)
        kmeans.train(x_learn)
        centroids = kmeans.centroids

        print(f"Reading base data from {base_path}...")
        x_base = fvecs_read(base_path)

        print("Assigning base points to clusters...")
        _, I = kmeans.index.search(x_base, 1)
        assignments = I.ravel()

        output_dir = os.path.join(dataset_dir, f"{k}_{niter}")
        save_results(output_dir, centroids, assignments, x_base)
    else:
        print(f"Could not find {learn_path} or {base_path}.")


def partition_data_with_duplicates(data_dir, dataset, k, niter, tail_ratio=0.9):
    """
    Partitions data where points in the 'tail' (furthest from centroid)
    are assigned to their 2 closest clusters.
    """
    if dataset == "sift":
        dataset_dir = os.path.join(data_dir, "sift")
        learn_path = os.path.join(dataset_dir, "sift_learn.fvecs")
        base_path = os.path.join(dataset_dir, "sift_base.fvecs")
    else:
        print(f"Dataset {dataset} not supported.")
        return

    if not (os.path.exists(learn_path) and os.path.exists(base_path)):
        print(f"Could not find data files.")
        return

    # 1. Train K-Means
    print(f"Reading training data from {learn_path}...")
    x_learn = fvecs_read(learn_path)
    d = x_learn.shape[1]

    print(f"Training K-means with k={k}...")
    kmeans = faiss.Kmeans(d, k, niter=niter, verbose=True)
    kmeans.train(x_learn)
    centroids = kmeans.centroids

    # 2. Search for top 2 nearest neighbors
    print(f"Reading base data from {base_path}...")
    x_base = fvecs_read(base_path)

    print("Searching for 2 nearest clusters for overlap calculation...")
    # D: Distances (N, 2), I: Indices (N, 2)
    D, I = kmeans.index.search(x_base, 2)

    primary_assignments = I[:, 0]
    primary_distances = D[:, 0]

    # 3. Calculate Thresholds (90th percentile distance per cluster)
    print(f"Calculating {int(tail_ratio * 100)}th percentile thresholds for duplication...")
    cluster_thresholds = np.zeros(k)

    for c in range(k):
        # Get distances of points primarily assigned to cluster c
        dists_in_cluster = primary_distances[primary_assignments == c]
        if len(dists_in_cluster) > 0:
            cluster_thresholds[c] = np.percentile(dists_in_cluster, 100 * tail_ratio)
        else:
            cluster_thresholds[c] = float('inf')

    # 4. Identify points to duplicate
    # A point is duplicated if: dist(point, primary_centroid) > threshold[primary_cluster]
    thresholds_per_point = cluster_thresholds[primary_assignments]
    # mask_duplicate[i] is True if point i is in the tail of its primary cluster
    mask_duplicate = primary_distances > thresholds_per_point

    num_duplicated = np.sum(mask_duplicate)
    print(f"identified {num_duplicated} points ({num_duplicated / len(x_base):.2%}) to duplicate to 2nd cluster.")

    # 5. Save Results
    output_dir = os.path.join(dataset_dir, f"{k}_{niter}_overlapping")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    centroids_path = os.path.join(output_dir, "centroids.fvecs")
    fvecs_write(centroids_path, centroids)
    print(f"Saved centroids to {centroids_path}")

    partitions_dir = os.path.join(output_dir, "partitions")
    if not os.path.exists(partitions_dir):
        os.makedirs(partitions_dir)

    all_ids = np.arange(x_base.shape[0], dtype=np.int64)

    print(f"Saving overlapping partitions to {partitions_dir}...")
    for c in range(k):
        # 1. Points that chose 'c' as their primary choice (Standard assignment)
        mask_primary = (primary_assignments == c)

        # 2. Points that chose 'c' as their secondary choice AND were outliers in their primary
        # Logic: Is 'c' the 2nd neighbor? AND Was the point an outlier in its 1st neighbor?
        mask_secondary = (I[:, 1] == c) & mask_duplicate

        # Combine
        final_mask = mask_primary | mask_secondary

        partition_data = x_base[final_mask]
        partition_ids = all_ids[final_mask]

        fvecs_write(os.path.join(partitions_dir, f"partition_{c}.fvecs"), partition_data)
        ids_write(os.path.join(partitions_dir, f"partition_{c}.ids"), partition_ids)

    print("Overlapping partitions saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    parser_download = subparsers.add_parser("download", help="Download dataset")
    parser_download.add_argument("--data_dir", type=str, default="data", help="Directory for data")
    parser_download.add_argument("--dataset", type=str, default="sift", help="Dataset name (default: sift)")

    # Partition command (Standard)
    parser_partition = subparsers.add_parser("partition", help="Standard K-means partition")
    parser_partition.add_argument("--data_dir", type=str, default="data")
    parser_partition.add_argument("--dataset", type=str, default="sift")
    parser_partition.add_argument("--k", type=int, default=100)
    parser_partition.add_argument("--niter", type=int, default=20)

    # Partition Duplicates command (New)
    parser_dupe = subparsers.add_parser("partition_duplicates", help="Partition with overlapping boundaries")
    parser_dupe.add_argument("--data_dir", type=str, default="data")
    parser_dupe.add_argument("--dataset", type=str, default="sift")
    parser_dupe.add_argument("--k", type=int, default=100)
    parser_dupe.add_argument("--niter", type=int, default=20)
    parser_dupe.add_argument("--ratio", type=float, default=0.9, help="Percentile threshold for tail (default 0.9)")

    args = parser.parse_args()

    if args.command == "download":
        download_data(args.data_dir, args.dataset)
    elif args.command == "partition":
        partition_data(args.data_dir, args.dataset, args.k, args.niter)
    elif args.command == "partition_duplicates":
        partition_data_with_duplicates(args.data_dir, args.dataset, args.k, args.niter, args.ratio)
    else:
        parser.print_help()
import numpy as np
import argparse
import os


def read_fvecs(fname):
    """
    Read an fvecs file.
    The format is: for each vector, 4 bytes (int32) for dimension d, 
    followed by d * 4 bytes (float32) for the vector components.
    """
    print(f"Reading fvecs file: {fname}")
    a = np.fromfile(fname, dtype='int32')
    if a.size == 0:
        print("File is empty")
        return np.array([])
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view('float32')


def read_ivecs(fname):
    """
    Read an ivecs file.
    The format is: for each vector, 4 bytes (int32) for dimension d,
    followed by d * 4 bytes (int32) for the vector components.
    """
    print(f"Reading ivecs file: {fname}")
    a = np.fromfile(fname, dtype='int32')
    if a.size == 0:
        print("File is empty")
        return np.array([])
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def read_npy(fname):
    """
    Read a npy file.
    """
    print(f"Reading npy file: {fname}")
    return np.load(fname)


def read_ids(fname):
    """
    Read an ids file (assumed int64).
    """
    print(f"Reading ids file: {fname}")
    return np.fromfile(fname, dtype=np.int64)


def inspect_file(fname, limit=5):
    if not os.path.exists(fname):
        print(f"File not found: {fname}")
        return

    ext = os.path.splitext(fname)[1]

    if ext == '.fvecs':
        data = read_fvecs(fname)
    elif ext == '.ivecs':
        data = read_ivecs(fname)
    elif ext == '.npy':
        data = read_npy(fname)
    elif ext == '.ids':
        data = read_ids(fname)
    else:
        print(f"Unknown extension: {ext}. Supported extensions: .fvecs, .ivecs, .npy, .ids")
        return

    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    if data.size > 0:
        print(f"  First {limit} elements/rows:\n{data[:limit]}")


def inspect_file_data(fname):
    if not os.path.exists(fname):
        print(f"File not found: {fname}")
        return

    ext = os.path.splitext(fname)[1]

    if ext == '.fvecs':
        data = read_fvecs(fname)
    elif ext == '.ivecs':
        data = read_ivecs(fname)
    elif ext == '.npy':
        data = read_npy(fname)
    elif ext == '.ids':
        data = read_ids(fname)
    else:
        print(f"Unknown extension: {ext}. Supported extensions: .fvecs, .ivecs, .npy, .ids")
        return
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and inspect .fvecs, .ivecs, .npy, and .ids files")
    parser.add_argument("file", type=str, help="Path to the file to read")
    parser.add_argument("--limit", type=int, default=5, help="Number of elements/rows to display (default: 5)")
    args = parser.parse_args()

    inspect_file(args.file, args.limit)
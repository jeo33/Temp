import numpy as np
import argparse
import os
from read_data import *
import os

print(os.getcwd())
dir=f'..\\data\\sift\\1000_20\\partitions'
groud_dir=f'..\\data\\sift\\sift_groundtruth.ivecs'
base=f'..\\data\\sift\\sift_base.fvecs'
id=[]
g = inspect_file_data(groud_dir)
b = inspect_file_data(base)

for it in range(1000):
    vecs = os.path.join(dir, f'partition_{it}.fvecs')
    ids = os.path.join(dir, f'partition_{it}.ids')
    v = inspect_file_data(vecs)
    i = inspect_file_data(ids)

    if not np.array_equal(b[i], v):
        print(f"Partition {it}: mismatch found")
    else:
        print(f"Partition {it}: OK")

print("good")




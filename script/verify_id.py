import numpy as np
import argparse
import os
from read_data import *

dir=f'..\\data\\sift\\1000_20\\partitions'
id=[]
for it in range(1000):
    fname=os.path.join(dir,f'partition_{it}.ids')
    ext = os.path.splitext(fname)[1]

    if ext == '.fvecs':
        data = read_fvecs(fname)
    elif ext == '.npy':
        data = read_npy(fname)
    elif ext == '.ids':
        data = read_ids(fname)
    id.extend(data)

if sum(id)==(0+999999)*1000000/2:
    print("data good")


#!/bin/bash

# Loop over different numbers of iterations
for niter in 10 20 50 100 200 400
do
    echo "Running partition with k=1000 and niter=$niter"
    python main.py partition --k 1000 --niter $niter
done
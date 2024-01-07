#!/bin/bash -x
mpirun -n 21 --oversubscribe build/task1
rm -rf fs

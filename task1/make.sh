#!/bin/bash -x

sudo rm -rf build/
mkdir build
cd build
cmake ..
make

cd ..
echo "Initial text" > file

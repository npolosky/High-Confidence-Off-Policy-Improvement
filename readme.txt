This library implements a Seldonian Optimization Algorithm for High Confidence Off-Policy Improvement and was completed as the final project for CMPSCI 687: Reinforcement Learning, taught by Phil Thomas at UMass Amherst.

The code depends on the Eigen, Boost, and OpenMP libraries. Eigen and Boost directories should be place in a folder called lib/ which resides in the same directory as the Makefile.

The data.csv file should go in the data/ folder.

The code that I have written is contained in the following files:

header/FCHC.hpp
header/FnApproxSoftmax.hpp
header/PDIS.hpp
header/Policy.hpp
header/TabularSoftmax.hpp
src/FCHC.cpp
src/FnApproxSoftmax.cpp
src/PDIS.cpp
src/TabularSoftmax.cpp
src/main.cpp

Other source and header files found in this directory may have been adapted but we're not originally wirtten by me. They were either provided for CMPSCI 687 Homework 4 for taken from Phil Thomas' AISafety Website.

To build the code run: 

make 

in the source directory.

To run the code run:

./main

in the source directory.
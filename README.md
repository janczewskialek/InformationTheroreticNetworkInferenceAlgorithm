# Information-Theoretic Network Inference Algorithm

**Author:** Aleksander Janczewski  
**Date:** 27th June 2024  
**Version:** 2.1

## Overview
This project offers Information-Theoretic Network Inference Algorithm

### /TransferEntropy

The C++ source code for an information-theoretic network inference algorithm. This algorithm leverages the KSG algorithm I for estimating continuous apparent and conditional transfer entropies based on the methodologies of Kraskov et al. [2004], Frenzel and Pompe [2007], Ragwitz and Kantz [2002] and Wibral et al. [2013].


## Features
- **KSG Algorithm I**: Continuous apparent and conditional transfer entropy estimations.
- **Embedding and Delays**: Derived using the Ragwitz criterion (Ragwitz and Kantz [2002]).
- **True Delay Detection**: Based on the methodology of Wibral et al. [2013].
- **Permutation Testing**: For statistical significance assessment.

## Requirements

### C++ Dependencies:
- C++17
- Clang 13.0.0 / GCC 11.2.0
- Eigen 3.4.0 (with unsupported Matrix functions module)
- CMake 3.2.0

## Directory Structure

| Directory | Description |
| --- | --- |
| /TransferEntropy | Main source code for information-theoretic network inference |
| /TransferEntropy/INA.cpp | Main implementation of the network inference algorithm |
| /TransferEntropy/ckdtree | Modified [Scipy's ckdtree](https://github.com/scipy/scipy/tree/main/scipy/spatial/ckdtree) |
| /circ_shift.h | [External function](https://stackoverflow.com/questions/46077242/eigen-modifyable-custom-expression/46301503#46301503) for array rolling |

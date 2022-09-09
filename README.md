# HPC Lab Project: Single Linkage Hierarchical Clustering
Repository for the final project of the course "High Perfomance Computing" [CM0227] held by Prof. Claudio Lucchese
at Ca' Foscari University of Venice during the first semester of the academic year 2021/2022.

As per specification, the source code files which contains the implementation of the slink algorithm and the parallelized
versions thereof can be compiled using c++11. However, due to the comfort offered by std::make_unique() the support libraries
such as Matrix and the CSV reader must be compiled using c++14. Technically speaking, it should be compliant with the assignment.

In order to enable debug output and result output, one can add to compile.sh the flag `-D DEBUG` to print out all the
operations performed by the algorithm to see how it works and the flag `-D DEBUG_OUT` to display the final result.

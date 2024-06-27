cmake_minimum_required(VERSION 3.20)
project(TransferEntropy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_library(ckdtree
        ckdtree/src/build.cxx
        ckdtree/src/count_neighbors.cxx
        ckdtree/src/query.cxx
        ckdtree/src/query_ball_point.cxx
        ckdtree/src/query_ball_tree.cxx
        ckdtree/src/query_pairs.cxx
        ckdtree/src/sparse_distances.cxx

        ckdtree/src/ckdtree_decl.h
        ckdtree/src/coo_entries.h
        ckdtree/src/distance.h
        ckdtree/src/distance_base.h
        ckdtree/src/ordered_pair.h
        ckdtree/src/rectangle.h
        )

add_executable(TransferEntropy
        main.cpp
        circ_shift.h
        INA.h
        INA.cpp
        )

add_library(TransferEntropyLib
        circ_shift.h
        INA.h
        INA.cpp
        )

target_link_libraries(TransferEntropy PUBLIC ckdtree)
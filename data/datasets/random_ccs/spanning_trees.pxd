

cdef packed struct Edge:
    int a
    int b

cdef packed struct LcaResult:
    int a
    int b
    int lca

cdef LcaResult* lowest_common_ancestor(int[:] parent, Edge[:] node_pairs);

cdef void calc_property_fast(int[:] parent, double[:] result, double root_val, int[:] degree, double (*update_fun)(int, int, double, int[:]));

cdef void random_spanning_tree_c(int size, int[:] degree, int** neighbors, int[:] parent, rnd);

cdef int** graph_to_neighbors(int size, int[:] degree, G);

cdef void free_graph_neighbors(int size, int** neighbors);
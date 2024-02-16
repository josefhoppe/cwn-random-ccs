# distutils: language=c++
# cython: profile=True
"""
Functions to 
"""

from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
#cimport numpy as np
import networkx as nx
import numpy as np
import cython
from typing import Tuple

cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937() # we need to define this constructor to stack allocate classes in Cython
        mt19937(unsigned int seed) # not worrying about matching the exact int type for seed
        unsigned int operator()()
    
    cdef cppclass uniform_int_distribution[T]:
        uniform_int_distribution()
        uniform_int_distribution(T a, T b)
        T operator()(mt19937 gen) # ignore the possibility of using other classes for "gen"


cdef packed struct Edge:
    int a
    int b

NP_EDGE = np.dtype([
    ('a', np.int32),
    ('b', np.int32),
])

cdef packed struct LcaResult:
    int a
    int b
    int lca

NP_LCA_RESULT = np.dtype([
    ('a', np.int32),
    ('b', np.int32),
    ('lca', np.int32),
])

cdef struct LcaLookup:
    int other
    int a
    int b

cdef inline int* uf_init(int size) nogil:
    """
    Initializes union-find datastructure with size many elements, each in their own partition
    """
    cdef int* ancestor = <int*> malloc(size * sizeof(int))
    cdef int i
    for i in range(size):
        ancestor[i] = i
    return ancestor

cdef inline int uf_find(int* parent, int item) nogil:
    """
    Find operation of union-find datastructure.
    parent has to be a valid array obtained through union and find operations, starting
    with a valid initialized array (e.g. uf_init)
    """
    cdef int partition = item
    # find correct root ancestor
    while parent[partition] != partition:
        partition = parent[partition]
    # update all ancestors to point to root
    p = item
    while p != partition:
        next_p = parent[p]
        parent[p] = partition
        p = next_p
    return partition

cdef inline void uf_union(int* parent, int a, int b) noexcept nogil:
    """
    unites the set containing a with the set containing b
    """
    parent[uf_find(parent, a)] = uf_find(parent, b)


cdef int __inner_lca(int node, vector[int]** children, vector[LcaLookup]** queries, LcaResult* result, int* result_count, int* partition, int* ancestor, char* node_color) nogil:
    for child in children[node][0]:
        __inner_lca(child, children, queries, result, result_count, partition, ancestor, node_color)
        uf_union(partition, node, child)
        ancestor[uf_find(partition, node)] = node
    node_color[node] = 1
    for lookup in queries[node][0]:
        other = lookup.other
        a = lookup.a
        b = lookup.b
        if node_color[other]:
            result[result_count[0]] = LcaResult(a, b, ancestor[uf_find(partition, other)])
            result_count[0] += 1
    return 0

def lowest_common_ancestor_py(parent: list, node_pairs: list) -> list[tuple]:
    c_parent = np.array(parent, dtype=np.int32)
    c_node_pairs = np.array(node_pairs, dtype=NP_EDGE)
    result = []
    c_res = lowest_common_ancestor(c_parent, c_node_pairs)
    cdef LcaResult lca_res
    for i in range(len(node_pairs)):
        lca_res = c_res[i]
        result.append((lca_res.a, lca_res.b, lca_res.lca))
    free(c_res)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef LcaResult* lowest_common_ancestor(int[:] parent, Edge[:] node_pairs):
    """
    Implementation of Tarjan's off-line lowest common ancestors algorithm
    (https://en.wikipedia.org/wiki/Tarjan%27s_off-line_lowest_common_ancestors_algorithm, https://doi.org/10.1145%2F322154.322161)

    Returns set of tuples (node a, node b, ancestor)

    The time complexity is in O(n + m * α(n + m)) for n = |nodes|, m = |node_pairs|
    (α is the inverse of the Ackermann function f(x) = A(x,x))

    Note that Tarjan later gave an improved algorithm with complexity O(n+m).
    Since the Ackermann function grows so fast, α is almost constant.
    It is especially faster than the sorting we perform afterward.
    """
    cdef int p_size = parent.shape[0]
    cdef char* node_color = <char*> malloc(p_size * sizeof(char))
    cdef int* ancestor = <int*> malloc(p_size * sizeof(int))
    cdef int i, node, p # both for iterating
    cdef Edge edge
    cdef int* partition = uf_init(p_size)

    cdef LcaResult* result = <LcaResult*> malloc(node_pairs.shape[0] * sizeof(LcaResult))
    cdef int result_count = 0
    cdef vector[int]** children = <vector[int]**> malloc(p_size * sizeof(void*))
    cdef vector[LcaLookup]** queries = <vector[LcaLookup]**> malloc(p_size * sizeof(void*))
    
    for i in range(p_size):
        node_color[i] = 0
        ancestor[i] = i
        children[i] = new vector[int]()
        queries[i] = new vector[LcaLookup]()


    try:
        # need to traverse the spanning tree top-down
        root = -1
        for node in range(p_size):
            p = parent[node]
            if p != -1:
                children[p][0].push_back(node)
            else:
                root = node

        # need to efficiently retrieve queries
        # node -> (other, edge[0], edge[1])
        #queries: list[list[tuple[int,int,int]]] = [ [] for _ in range(parent.shape[0])]
        #queries = np.empty(parent.shape[0], dtype=object)

        for i in range(node_pairs.shape[0]):
            edge = node_pairs[i]
            a = edge.a
            b = edge.b
            queries[a][0].push_back(LcaLookup(b,a,b))
            queries[b][0].push_back(LcaLookup(a,a,b))
        
        __inner_lca(root, children, queries, result, &result_count, partition, ancestor, node_color)

    finally:
        free(ancestor)
        free(partition)
        free(node_color)
        for i in range(p_size):
            del children[i]
            del queries[i]
        free(children)
        free(queries)

    return result

def normalize_cell(cell: tuple) -> tuple:
    """
    normalizes tuple to have the smallest element at index 0 and the smaller
    connected one at index 1 (instead of -1)
    """
    min_index = cell.index(min(cell))
    shifted = cell[min_index:] + cell[:min_index]
    if shifted[-1] < shifted[1]:
        shifted = shifted[::-1]
        shifted = shifted[-1:] + shifted[:-1]
    return shifted

def get_induced_cycle(edge: Tuple[int, int], parent: np.ndarray, depth: np.ndarray) -> tuple:
    """
    Gets the cycle induced by adding edge to the spanning tree modeled by node_level and parent_node
    """
    left = []
    right = []

    a = edge[0]
    b = edge[1]

    if depth[a] < depth[b]:
        a = edge[1]
        b = edge[0]

    while depth[a] > depth[b]:
        left.append(a)
        a = parent[a]

    while a != b:
        left.append(a)
        right.append(b)
        a = parent[a]
        b = parent[b]

    left.append(a)
    return normalize_cell(tuple(left + right[::-1]))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int __calc_depth_check(int node, int[:] parent, int[:] depth):
    if node != -1 and depth[node] == -1:
        if parent[node] == -1:
            depth[node] == 0
        else:
            __calc_depth_check(parent[node], parent, depth)
            depth[node] = depth[parent[node]] + 1
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_depth(parent: np.ndarray) -> np.ndarray:
    depth = -1 * np.ones_like(parent)
    cdef int[:] cdepth = depth
    cdef int[:] cparent = parent

    for i in range(len(parent)):
        __calc_depth_check(i, cparent, cdepth)
    return depth

@cython.wraparound(False)
cdef void random_spanning_tree_c(int size, int[:] degree, int** neighbors, int[:] parent, rnd):
    cdef mt19937 c_rnd = mt19937(rnd.integers(0, 1 << 32))
    cdef uniform_int_distribution[int] dist = uniform_int_distribution[int](0, size - 1)
    cdef int root = dist(c_rnd) #rnd.choice(size) #rand() % size
    cdef int i, u, choice

    cdef bint* in_tree = <bint*> malloc(size * sizeof(bint))
    try:
        for i in range(size):
            in_tree[i] = 0

        in_tree[root] = 1
        parent[root] = -1

        for i in range(size):
            u = i
            while not in_tree[u]:
                dist = uniform_int_distribution[int](0, degree[u] - 1)
                choice = dist(c_rnd) #c_rnd() % degree[u] # random(c_rnd, degree[u]) #rand() % degree[u]#rnd.choice(degree[u]) # 
                parent[u] = neighbors[u][choice]
                u = parent[u]
            
            u = i
            while not in_tree[u]:
                in_tree[u] = 1
                u = parent[u]
    finally:
        free(in_tree)

@cython.wraparound(False)
cdef int** graph_to_neighbors(int size, int[:] degree, G):
    cdef int** neighbors = <int**> malloc(sizeof(int*) * size)
    for i in range(size):
        neighbors[i] = <int*> malloc(degree[i] * sizeof(int))
        for j, u in enumerate(G[i]):
            neighbors[i][j] = u
    return neighbors

cdef void free_graph_neighbors(int size, int** neighbors):
    for i in range(size):
        free(neighbors[i])
    free(neighbors)

def random_spanning_tree(G: nx.Graph, rnd: np.random.Generator) -> np.ndarray[np.int32]:
    """
    Implements Wilson's Algorithm for random spanning trees [1].
    Assumes G to be undirected and connected.

    [1] David Bruce Wilson. 1996. Generating random spanning trees more quickly than the cover time. In Proceedings of the twenty-eighth annual ACM symposium on Theory of Computing (STOC '96). Association for Computing Machinery, New York, NY, USA, 296–303. https://doi.org/10.1145/237814.237880
    """
    cdef int size = len(G.nodes)
    cdef int[:] degree = np.array(G.degree, dtype=np.int32)[:,1]
    np_parent = np.zeros(size, dtype=np.int32)
    cdef int[:] parent = np_parent
    neighbors = graph_to_neighbors(size, degree, G)

    try:
        #srand(rnd.choice(100) + 5)
        random_spanning_tree_c(size, degree, neighbors, parent, rnd)
    finally:
        free_graph_neighbors(size, neighbors)
    
    return np_parent

@cython.wraparound(False)
cdef void __calc_property_check(int node, int[:] parent, char[:] checked, double[:] result, double root_val, int[:] degree, double (*update_fun)(int, int, double, int[:])):
    if node != -1:
        p = parent[node]
        if p != -1 and checked[node] == 0:
            if checked[p] == 0:
                __calc_property_check(p, parent, checked, result, root_val, degree, update_fun)
            result[node] = update_fun(node, p, result[p], degree)
            checked[node] = 1
        if p == -1 and checked[node] == 0:
            result[node] = root_val

@cython.wraparound(False)
cdef void calc_property_fast(int[:] parent, double[:] result, double root_val, int[:] degree, double (*update_fun)(int, int, double, int[:])):
    checked_np = np.zeros(len(parent), dtype=np.int8)
    cdef char[:] checked = checked_np

    for i in range(len(parent)):
        __calc_property_check(i, parent, checked, result, root_val, degree, update_fun)
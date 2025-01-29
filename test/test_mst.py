import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """
    #check that weight matches expected weight of MST
    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    #check that MST spans all nodes
    connected_nodes = set()
    for i in range(len(mst)):
        for j in range(len(mst)):
            if mst[i, j] > 0:  #if there is an edge between i and j, add nodes to set of connected nodes
                connected_nodes.add(i)
                connected_nodes.add(j)
    assert len(connected_nodes) == len(adj_mat), "MST does not span all nodes!"

    #check that the MST has v-1 edges
    #count number of edges in MST: 
    num_edges = sum(sum(mst > 0)) // 2   #sum across rows and then sum across columns
    assert num_edges == len(mst) - 1, "MST has incorrect number of edges"



def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    
    TODO: Write at least one unit test for MST construction.
    
    """

    #test edge cases

    #test on a graph with only one node
    adj_mat = np.array([[0]])
    single_node_graph = Graph(adj_mat)
    single_node_graph.construct_mst()
    check_mst(adj_mat, single_node_graph.mst, 0)

    #test on a graph with two nodes
    adj_mat = np.array([[0, 1], [1, 0]])
    two_node_graph = Graph(adj_mat)
    two_node_graph.construct_mst()
    check_mst(adj_mat, two_node_graph.mst, 1)

    #test on a graph with all zero weights
    adj_mat = np.zeros((5,5))
    zero_weight_graph = Graph(adj_mat)
    zero_weight_graph.construct_mst()
    check_mst(adj_mat, zero_weight_graph.mst, 0)
    

    #test_graph = Graph("data/student_test.csv") #choose more interesting example than a random small
    #test_graph.construct_mst()
    #check_mst(test_graph.adj_mat, test_graph.mst, 4) # fails. what is the expected weight of the MST? 
    pass

import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        
        '''
        pseudocode for Prim's algorithm, minimum spanning tree- from lecture slides
        PRIM (V, E, c)
        S ←∅, T←∅.
        s ← any node in V.
        FOREACHv≠s: π[v] ← ∞,pred[v] ← null; π[s] ← 0. Create an empty priority queue pq.
        FOREACH v ∈ V : INSERT(pq, v, π[v]).
        WHILE (IS-NOT-EMPTY(pq))
        u ← DEL-MIN(pq).
        S ← S ∪ { u }, T ← T ∪ { pred[u] }.
        FOREACH edge e = (u, v) ∈ E with v ∉ S : IF (ce < π[v])
        DECREASE-KEY(pq, v, ce). π[v] ← ce; pred[v] ← e.
    
        '''
        

        #initialize mst adjancency matrix
        self.mst = np.zeros_like(self.adj_mat)
        
        #nodes in mst so far
        mst_nodes = set () #S
        
        #edges in mst so far
        mst_edges = [] #T

        #choose any random node in graph to start with
        start_node = 0

        #initialize pi and pred
        minimum_cost_to_node_dict = {node: float('inf') for node in range(len(self.adj_mat))} #pi
        predecessor_dict = {node: None for node in range(len(self.adj_mat))} #pred

        #minimum cost to start node is 0
        minimum_cost_to_node_dict[start_node] = 0

        #create empty priority queue using heapq
        priority_queue = [] #pq
        heapq.heapify(priority_queue)

        #store tuples of (priority, node) to maintain the priority.
        for node in range(len(self.adj_mat)):
            heapq.heappush(priority_queue, (minimum_cost_to_node_dict[node], node))
        
        #while priority queue is not empty
        while priority_queue:
            #get node with minimum cost
            current_node = heapq.heappop(priority_queue)[1]

            #if current node is already in mst, skip
            if current_node in mst_nodes:
                continue

            #add node to mst
            mst_nodes.add(current_node)

            #if predecessor exists, add to mst_edges
            if predecessor_dict[current_node] is not None:
                mst_edges.append(predecessor_dict[current_node])

            #iterate over all edges of current node, check cost and update pi and pred accordingly
            #for each edge 
            for node, edge_weight in enumerate(self.adj_mat[current_node]): 
                #if node is not already in mst and edge is not zero:
                if node not in mst_nodes and edge_weight != 0:
                    #if edge weight is less than current minimum cost to node (node or current_node?)
                    if edge_weight < minimum_cost_to_node_dict[node]:
                        #update cost to node
                        minimum_cost_to_node_dict[node] = edge_weight
                        #update predecessor of node
                        predecessor_dict[node] = (current_node, node)
                        #update priority queue
                        heapq.heappush(priority_queue, (minimum_cost_to_node_dict[node], node))
        
        #convert mst_edges to mst adjacency matrix
        for u, v in mst_edges:
            weight = self.adj_mat[u, v] #get weight of edge between node u and v from adjacency matrix
            self.mst[u, v] = weight
            self.mst[v, u] = weight  #symmetric matrix
        
############################################

def main(self):
    print(self.adj_mat)
    #random_ix = np.random.choice(len(self.adj_mat))
    #print(self.adj_mat[random_ix])
    #print(self.adj_mat[0][1])
    self.construct_mst()
    print("MST:")
    print(self.mst)
 
 #checking 
    connected_nodes = set()
    for i in range(len(self.mst)):
        for j in range(len(self.mst)):
            if self.mst[i, j] > 0:  # If there's an edge between i and j
                connected_nodes.add(i)
                connected_nodes.add(j)
    if len(connected_nodes) != len(self.adj_mat):
        print("MST does not span all nodes!")
    else:
        print("MST spans all nodes!")


if __name__ == "__main__":
    #create a graph object
    graph = Graph("data/small.csv")
    main(graph)
    
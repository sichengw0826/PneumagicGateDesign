import networkx as nx
import matplotlib.pyplot as plt

edge_types = ['coupled','opposite','pneumatic']

def subgraphs_by_edge_type(G, edge_type):
    """
    Given a graph G and a desired edge_type (string), return a list of
    connected subgraphs of G, each induced by edges whose 'type' == edge_type.
    """
    # 1. Filter edges matching the desired type
    matching_edges = [
        (u, v) for u, v, attrs in G.edges(data=True)
        if attrs.get('type') == edge_type
    ]
    
    # 2. Create the subgraph induced by these edges (includes only nodes incident to them)
    H = G.edge_subgraph(matching_edges).copy()
    
    # 3. Break H into its connected components and return them as separate graphs
    subgraphs = []
    for component in nx.connected_components(H):
        # component is a set of nodes; induce a subgraph on them
        subg = H.subgraph(component).copy()
        subgraphs.append(subg)
    
    return subgraphs

AND = nx.Graph()
#AND.add_nodes_from(['sup','no1','no2','nc1','nc2','out','gnd'])
AND.add_edges_from([('sup','nc1'),('nc1','nc2'),('nc2','out'),
                   ('out','no1'),('out','no2'),('no1','gnd'),('no2','gnd')],type=edge_types[2])
AND.add_edges_from([('nc1','no1'),('nc2','no2')],type=edge_types[1])

XOR = nx.Graph()
XOR.add_nodes_from(['sup','no1','no2','nc1','nc2','out','gnd'])

ax = plt.subplot()
nx.draw(AND,with_labels=True)
plt.show()
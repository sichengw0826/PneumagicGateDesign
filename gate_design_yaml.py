import yaml
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import OrigamiPiece
from matplotlib.lines import Line2D
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph

def parse_graph_from_yaml(data):
    """
    Load a YAML file with keys:
      - 'nodes': list of { 'name' or 'ID', 'num': [..], 'type': str }
      - any keys starting with 'edges_': each a list of { 'source': , 'target': }
    
    Returns a NetworkX Graph with:
      - node attribute 'type'
      - edge attribute 'type' (the YAML key, e.g. 'edges_pneumatic')
    """

    G = nx.Graph()
    # --- add nodes ---
    for entry in data.get('nodes', []):
        base = entry.get('name', [])
        for num in entry.get('num', []):
            node_name = f"{base}{num}"
            G.add_node(node_name, type=entry.get('type'))

    # --- add edges ---
    for section, elist in data.items():
        if section.startswith('edges_'):
            for edge in elist:
                src = edge['source']
                tgt = edge['target']
                G.add_edge(src, tgt, type=section)

    return G

def edge_predecessors_successors(G, edge):
    """
    Given a graph G and an edge tuple (u, v), return two lists:
      - preds: all edges “before” (into u if directed, or incident on u if undirected)
      - succs: all edges “after” (out of v if directed, or incident on v if undirected)
    The given edge itself is excluded from both lists.
    """
    u, v = edge

    if G.is_directed():
        # predecessors are incoming to u (excluding the edge itself)
        preds = [(x, u, d) for x, _, d in G.in_edges(u, data=True)
                 if (x, u) != (u, v)]
        # successors are outgoing from v
        succs = [(v, w, d) for _, w, d in G.out_edges(v, data=True)
                  if (v, w) != (u, v)]
    else:
        # undirected: “before” = all edges incident to u except (u,v)
        preds = [(u, nbr, G.edges[u, nbr]) 
                 for nbr in G.neighbors(u) if nbr != v]
        # “after” = all edges incident to v except (u,v)
        succs = [(v, nbr, G.edges[v, nbr])
                  for nbr in G.neighbors(v) if nbr != u]

    return preds, succs

def computeOrigamiHeight(fold_angle):
    h_mid = 15  #placeholder, mm
    h_flap = 7
    h_padding = 3
    h_connector = 16.8

    return h_mid, h_flap, h_padding,h_connector


if __name__ == "__main__":
    # parse
    yaml_file = "XOR_def.yaml"    # adjust to your actual path
    with open(yaml_file, 'r') as f:
        gate_data = yaml.safe_load(f)
    G = parse_graph_from_yaml(gate_data)
    base_channel_width = gate_data.get('base_channel_width',[])
    fold_angle = gate_data.get('fold_angle',[])
    u_turn_min_sep = gate_data.get('u_turn_min_sep',[])
    default_padding = gate_data.get('default_padding',[])
    IO_side = gate_data.get('IO_side',[])  # Use following definition: assume normally opened side on the top, 1 means supply is on the right, 0 therwise
    linkage_dist = gate_data.get('linkage_dist',[])

    # pick only the coupled/opposite edges, i.e., the description of the mechanical structure of the gate
    mechanical_edge_types = {'edges_coupled', 'edges_opposite'}
    skeleton_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get('type') in mechanical_edge_types
    ]
    
    origami_skeleton = G.edge_subgraph(skeleton_edges)  # build the subgraph induced by those edges

    num_joined_graphs = nx.number_connected_components(origami_skeleton)    # count how many connected components it has

    print(f"origami_skeleton has {num_joined_graphs} joined graph(s).")

    mechanical_pairs_list = [origami_skeleton.subgraph(linked) for linked in nx.connected_components(origami_skeleton)]

    # Determine the width of the flaps by counting the channels crossing the fold; compute fold crossing location
    offset_list = [-1, 1]
    for pair, offset_sign in zip(mechanical_pairs_list,offset_list):  # iterate over each 4-bar linkage unit
        center_piece_edge = next((u,v) for u,v,d in pair.edges(data=True) if d.get('type') == 'edges_opposite') # edge denoting the fixed origami piece
        coupled_list = [(u,v) for u,v,d in pair.edges(data=True) if d.get('type') == 'edges_coupled']
        sides = pair.edge_subgraph(coupled_list)
        computed_widths = {}
        for flap in nx.connected_components(sides): # break the linkage into the normally closed and normally opened sides, then loop through nodes in each flap
            ch_count = 0
            for n in flap:
                n_type = G.nodes[n].get('type')
                # iterate over all edges incident to n
                for _, _, edata in G.edges(n, data=True):
                    if edata.get('type') == 'edges_pneumatic':
                        # subtract 1 if this node is a GND, else add 1
                        ch_count += -1 if n_type == 'GND' else 1
                if n_type == 'FOLD_NC':
                    dict_str = 'nc_w'
                elif n_type == 'FOLD_NO':
                    dict_str = 'no_w'
            flap_width = ch_count*base_channel_width+u_turn_min_sep+default_padding
            computed_widths[dict_str] = flap_width
        
        flapWidth = max(computed_widths.values())

        # compute the height (length) of each part of the gate
        h_mid, h_flap, h_padding,h_connector = computeOrigamiHeight(fold_angle)

        # create origami object and write to the graph
        blockObj = OrigamiPiece.OrigamiPiece(center_piece_edge,width=flapWidth,height=h_mid,centroidPos=(linkage_dist+flap_width)/2*offset_sign)
        pair[center_piece_edge[0]][center_piece_edge[1]]['block']=blockObj

        # compute the location of fold-crossing and input location
        def compute_crossing_pos(nodeType):
            if nodeType == 'FOLD_NO':
                y_sign = 1
            elif nodeType == 'FOLD_NC':
                y_sign = -1
            crossing_list = [n for n,d in pair.nodes(data=True) if d.get('type') == nodeType] # first, get all nodes of the same type within the 4-bar linkage unit
            print(blockObj.centroidPos)
            x_pos = [-flapWidth/2+ flapWidth/(len(crossing_list)+1)*k for k in range(1, len(crossing_list)+1)]
            y_pos = h_mid/2*y_sign
            for k in range(0,len(crossing_list)):
                pair.nodes[crossing_list[k]]['local_pos'] = (x_pos[k],y_pos)
                pair.nodes[crossing_list[k]]["pos"] = f"{abs(int(x_pos[k]+blockObj.centroidPos)*100)},{abs(int(y_pos*100))}!"
                pair.nodes[crossing_list[k]]["pin"] = "true"
        
        compute_crossing_pos('FOLD_NO')
        compute_crossing_pos('FOLD_NC')

        print(origami_skeleton.nodes.data())
    


            


    # 2. layout
    
    #pos = nx.spring_layout(G, seed=42)
    pos = graphviz_layout(G, prog="neato")#, args="-n2")
    print(pos)

    # 3. gather types
    node_types = sorted({d['type'] for _,d in G.nodes(data=True)})
    edge_types = sorted({d['type'] for _,_,d in G.edges(data=True)})

    # 4. build color/shape/style maps
    cmap = plt.cm.tab10.colors
    color_cycle = itertools.cycle(cmap)
    # assign each node‑ or edge‑type a distinct color
    color_map = { t: next(color_cycle) for t in node_types + edge_types }

    # node shapes
    node_shapes = ['o','s','^','d','p','h','8','v','<','>']
    shape_cycle = itertools.cycle(node_shapes)
    shape_map = { t: next(shape_cycle) for t in node_types }

    # edge line styles
    line_styles = ['solid','dashed','dotted','dashdot']
    style_cycle = itertools.cycle(line_styles)
    style_map = { t: next(style_cycle) for t in edge_types }

    # 5. draw
    plt.figure(figsize=(8,6))
    # draw nodes by type
    for nt in node_types:
        nlist = [n for n,d in G.nodes(data=True) if d['type']==nt]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nlist,
            node_shape=shape_map[nt],
            node_color=[color_map[nt]],
            label=nt,
            node_size=500
        )

    # draw edges by type
    for et in edge_types:
        elist = [(u,v) for u,v,d in G.edges(data=True) if d['type']==et]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=elist,
            style=style_map[et],
            edge_color=[color_map[et]],
            label=et,
            width=2
        )

    # draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)

    # custom legend
    legend_elements = []
    for nt in node_types:
        legend_elements.append(
            Line2D([0],[0],
                   marker=shape_map[nt],
                   color='w',
                   label=nt,
                   markerfacecolor=color_map[nt],
                   markersize=10)
        )
    for et in edge_types:
        legend_elements.append(
            Line2D([0],[0],
                   linestyle=style_map[et],
                   color=color_map[et],
                   label=et,
                   linewidth=2)
        )

    plt.legend(handles=legend_elements, loc='best')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
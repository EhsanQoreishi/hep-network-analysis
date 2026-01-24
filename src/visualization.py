import os
import networkx as nx
from pyvis.network import Network
from community import community_louvain

def visualize_network(G: nx.Graph, title: str = "results/hep_interactive_map.html") -> None:
    """
    Generates a high-fidelity interactive HTML visualization of the network topology.

    To ensure visual interpretability and performance, the function extracts the
    top 500 nodes by degree (hubs) and projects their connectivity. Nodes are
    scaled by their degree centrality and color-coded based on their detected
    Louvain community membership.
    
    Args:
        G (nx.Graph): The social network graph.
        title (str): Path where the HTML file will be saved.
    """
    print("\n--- Projecting Interactive Topological Map ---")

    # Ensure the directory exists
    output_dir = os.path.dirname(title)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Filter the network to focus on the top 500 hubs for visual clarity & performance
    degrees = dict(G.degree())
    # Sort nodes by degree and take top 500
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:500]
    G_sub = G.subgraph(top_nodes)

    print(f"  Rendering subgraph with {len(top_nodes)} nodes (Top Hubs)...")

    # Configure Pyvis: Dark theme for contrast
    net = Network(height="100vh", width="100%", bgcolor="#222222", font_color="white")

    # Apply modularity maximization on the SUBGRAPH to color-code strictly visible clusters
    partition = community_louvain.best_partition(G_sub)

    # Map graph nodes to the interactive environment
    for node in G_sub.nodes():
        comm_id = partition.get(node, 0)
        degree = degrees[node]
        
        # Add node with metadata tooltips
        net.add_node(
            node,
            label=node,
            title=f"Author: {node}\nDegree: {degree}\nCommunity: {comm_id}",
            value=degree, # Size depends on degree
            group=comm_id, # Color depends on community
        )

    # Map edges with neutral coloring
    for u, v in G_sub.edges():
        net.add_edge(u, v, color="#555555", alpha=0.3)

    # Apply ForceAtlas2 physics for a layout that emphasizes modular separation
    net.force_atlas_2based()

    # Save output
    try:
        net.save_graph(title)
        print(f"  Success! Interactive map saved to: {title}")
    except Exception as e:
        print(f"  Error saving visualization: {e}")
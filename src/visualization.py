import logging
import os

import networkx as nx
from community import community_louvain
from pyvis.network import Network

logger = logging.getLogger(__name__)


def visualize_network(
    G: nx.Graph, title: str = "results/hep_interactive_map.html"
) -> None:
    """
    Generates a high-fidelity interactive HTML visualization of the network topology.

    To ensure visual interpretability and performance, the function extracts the
    top 500 nodes by degree (hubs) and projects their connectivity. Nodes are
    scaled by their degree centrality and color-coded based on their detected
    Louvain community membership (recalculated for the subgraph).

    Args:
        G (nx.Graph): The social network graph.
        title (str): Path where the HTML file will be saved.
    """
    logger.info("--- Projecting Interactive Topological Map ---")

    # Ensure output directory exists
    output_dir = os.path.dirname(title)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Filter for Top Hubs (Top 500 by Degree)
    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:500]
    G_sub = G.subgraph(top_nodes)

    logger.info(f"  Rendering subgraph with {len(top_nodes)} nodes (Top Hubs)...")

    # Initialize PyVis Network
    net = Network(height="100vh", width="100%", bgcolor="#222222", font_color="white")

    # Detect communities within this subgraph for coloring
    partition = community_louvain.best_partition(G_sub)

    # Add nodes with visual attributes
    for node in G_sub.nodes():
        comm_id = partition.get(node, 0)
        degree = degrees[node]
        net.add_node(
            node,
            label=node,
            title=f"Author: {node}\nDegree: {degree}\nCommunity: {comm_id}",
            value=degree,
            group=comm_id,
        )

    # Add edges
    for u, v in G_sub.edges():
        net.add_edge(u, v, color="#555555", alpha=0.3)

    # Physics Layout
    net.force_atlas_2based()

    try:
        net.save_graph(title)
        logger.info(f"  Success! Interactive map saved to: {title}")
    except Exception as e:
        logger.error(f"  Error saving visualization: {e}")

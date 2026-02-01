import heapq
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

    Optimizations:
    - Uses Heap Queue (heapq) for O(N log K) top-node extraction instead of O(N log N) sorting.
    - Uses batch attribute assignment and `from_nx` to minimize Python loop overhead.

    Args:
        G (nx.Graph): The social network graph.
        title (str): Path where the HTML file will be saved.
    """
    logger.info("--- Projecting Interactive Topological Map ---")

    output_dir = os.path.dirname(title)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    degrees = dict(G.degree())
    k_hubs = 500

    if len(degrees) > k_hubs:
        top_nodes = heapq.nlargest(k_hubs, degrees, key=degrees.get)
    else:
        top_nodes = list(degrees.keys())

    G_sub = G.subgraph(top_nodes).copy()

    logger.info(f"  Rendering subgraph with {len(top_nodes)} nodes (Top Hubs)...")

    partition = community_louvain.best_partition(G_sub)

    titles = {}
    groups = {}
    values = {}

    for node in G_sub.nodes():
        comm_id = partition.get(node, 0)
        deg = degrees[node]

        titles[node] = f"Author: {node}\nDegree: {deg}\nCommunity: {comm_id}"
        groups[node] = comm_id
        values[node] = deg

    nx.set_node_attributes(G_sub, titles, "title")
    nx.set_node_attributes(G_sub, groups, "group")
    nx.set_node_attributes(G_sub, values, "value")
    nx.set_node_attributes(G_sub, {n: n for n in G_sub.nodes()}, "label")

    net = Network(height="100vh", width="100%", bgcolor="#222222", font_color="white")

    net.from_nx(G_sub)

    net.force_atlas_2based()

    for edge in net.edges:
        edge["color"] = "#555555"
        edge["alpha"] = 0.3

    try:
        net.save_graph(title)
        logger.info(f"  Success! Interactive map saved to: {title}")
    except Exception as e:
        logger.error(f"  Error saving visualization: {e}")

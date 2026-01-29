import logging
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def get_global_metrics(G: nx.Graph) -> Dict[str, float]:
    """
    Computes standard topological metrics.

    Metrics:
    - Density: Ratio of actual edges to possible edges.
    - Transitivity: Global clustering (fraction of closed triplets).
    - Avg Clustering: Mean of local clustering coefficients.

    Returns:
        Dict: Contains 'nodes', 'edges', 'density', 'transitivity', 'avg_clustering'.
    """
    logger.info("--- Global Graph Metrics ---")

    metrics = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "transitivity": nx.transitivity(G),
        "avg_clustering": nx.average_clustering(G),
    }

    logger.info(f"Nodes: {metrics['nodes']}, Edges: {metrics['edges']}")
    logger.info(f"Edge Density: {metrics['density']:.6f}")
    logger.info(
        f"Global Clustering Coeff (Transitivity): {metrics['transitivity']:.4f}"
    )
    logger.info(f"Average Clustering Coefficient: {metrics['avg_clustering']:.4f}")

    return metrics


def get_top_authors(
    G_co: nx.Graph, G_cit: nx.DiGraph
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Identifies top authors by Degree (Social) and In-Degree (Citation).
    Also calculates Betweenness Centrality to find 'bridges'.

    Returns:
        Dict: Lists of top authors for 'collaborative', 'influential', and 'bridges'.
    """
    logger.info("--- Centrality Analysis ---")

    # 1. Degree Centrality (Hubs)
    # Note: Sorting dict items is efficient enough here; no complex vectorization needed for top-k
    top_connected = sorted(G_co.degree(), key=lambda x: x[1], reverse=True)[:5]

    logger.info("Most Collaborative (High Degree):")
    for author, degree in top_connected:
        logger.info(f"  - {author}: {degree} co-authors")

    # 2. Citation Influence (Authorities)
    top_cited = sorted(
        G_cit.in_degree(weight="weight"), key=lambda x: x[1], reverse=True
    )[:5]

    logger.info("Most Influential (High Citations):")
    for author, count in top_cited:
        logger.info(f"  - {author}: {count:.1f} citations")

    # 3. Betweenness Centrality (Bridges)
    logger.info("Calculating Betweenness Centrality (this may take a moment)...")
    top_bridges = []

    try:
        # k=500 approximation for speed on large graphs
        betweenness = nx.betweenness_centrality(G_co, weight="distance", k=500)
        top_bridges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]

        logger.info("Top Bridges (High Betweenness):")
        for author, score in top_bridges:
            logger.info(f"  - {author}: {score:.4f}")

    except (KeyError, ValueError) as e:
        logger.error(f"Error calculating betweenness: {e}")

    return {
        "collaborative": top_connected,
        "influential": top_cited,
        "bridges": top_bridges,
    }


def analyze_layer_shortest_paths(
    G_cit: nx.DiGraph, G_co: nx.Graph, output_dir: str = "results"
) -> float:
    """
    Calculates the social distance (co-authorship path) between authors
    who have a direct citation link.
    """
    logger.info("--- Cross-Layer Path Analysis ---")

    # Pre-filter nodes to avoid repeated dictionary lookups in the loop
    social_nodes = set(G_co.nodes())

    # Identify valid citation edges where both nodes exist in social layer
    # Vectorization note: Shortest path is inherently sequential per pair,
    # but we filter the edge list first.
    valid_edges = [
        (u, v) for u, v in G_cit.edges() if u in social_nodes and v in social_nodes
    ]

    distances = []
    for u, v in valid_edges:
        try:
            d = nx.shortest_path_length(G_co, source=u, target=v, weight="distance")
            distances.append(d)
        except nx.NetworkXNoPath:
            continue

    avg_dist = np.mean(distances) if distances else 0.0

    logger.info(f"Analyzed {len(distances)} valid pairs connected in Citation layer.")
    logger.info(f"Average Co-authorship distance for these pairs: {avg_dist:.2f}")

    # Plot
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(
        distances,
        bins=50,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    plt.title("Social Distance of Cited Authors")
    plt.xlabel("Shortest Path Length (Co-authorship)")
    plt.ylabel("Frequency")
    plt.axvline(avg_dist, color="red", linestyle="dashed", label=f"Avg: {avg_dist:.2f}")
    plt.legend()

    save_path = os.path.join(output_dir, "cross_layer_path_distribution.png")
    plt.savefig(save_path)
    plt.close()

    return float(avg_dist)


def analyze_strength_distribution(
    G: nx.Graph, name: str = "Network", output_dir: str = "results"
) -> Dict[str, float]:
    """
    Analyzes the correlation between Node Strength (s) and Degree (k).
    s ~ k^beta.
    """
    logger.info(f"--- Weighted Strength Analysis ({name}) ---")

    # Extract metrics using list comprehensions (faster than dict iteration)
    degrees = dict(G.degree())
    strengths = dict(G.degree(weight="weight"))

    # Vectorize inputs
    nodes = list(G.nodes())
    k_values = np.array([degrees[n] for n in nodes])
    s_values = np.array([strengths[n] for n in nodes])

    # Boolean masking to filter zeros
    mask = (k_values > 0) & (s_values > 0)
    k_values = k_values[mask]
    s_values = s_values[mask]

    # Calculate average strength per degree class
    k_unique = np.unique(k_values)
    # Vectorized calculation of means per group is complex without pandas,
    # keeping the explicit loop over unique k is cleaner for pure numpy here.
    s_avg_k = np.array([np.mean(s_values[k_values == k]) for k in k_unique])

    # Fit Power Law: log(s) ~ beta * log(k) + intercept
    log_k = np.log10(k_unique)
    log_s = np.log10(s_avg_k)
    beta, intercept = np.polyfit(log_k, log_s, 1)

    logger.info(f"  Fit exponent (beta): {beta:.4f}")
    if beta > 1.1:
        logger.info("  -> Super-linear (beta > 1): Hubs work harder/more intensely.")

    # Plot
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(k_values, s_values, alpha=0.1, color="gray", s=10, label="Nodes")
    plt.loglog(k_unique, s_avg_k, "bo", label="Average <s(k)>")

    fit_y = 10**intercept * k_unique**beta
    plt.plot(k_unique, fit_y, "r--", linewidth=2, label=f"Fit: $\\beta = {beta:.2f}$")

    plt.title("Strength vs Degree Correlation")
    plt.xlabel("Degree k")
    plt.ylabel("Strength s")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(
        output_dir, f"{name.lower()}_strength_degree_correlation.png"
    )
    plt.savefig(save_path)
    plt.close()

    return {"beta": beta, "intercept": intercept}


def analyze_multiplex_correlation(
    G_co: nx.Graph, G_cit: nx.DiGraph, output_dir: str = "results"
) -> Dict[str, float]:
    """
    Correlates Citation Influence (PageRank) with Social Brokerage (Betweenness).
    """
    logger.info("--- Multiplex Correlation Analysis ---")

    common_authors = list(set(G_co.nodes()).intersection(set(G_cit.nodes())))
    if len(common_authors) < 10:
        logger.warning("Not enough common authors for correlation.")
        return {"correlation": 0.0, "p_value": 1.0}

    pagerank = nx.pagerank(G_cit, weight="weight")
    # k=min(...) ensures we don't over-sample small graphs
    k_sample = min(len(common_authors), 500)
    betweenness = nx.betweenness_centrality(G_co, weight="distance", k=k_sample)

    # Convert to arrays for vectorized filtering
    x_data = np.array([pagerank.get(a, 0) for a in common_authors])
    y_data = np.array([betweenness.get(a, 0) for a in common_authors])

    corr, p_value = spearmanr(x_data, y_data)

    logger.info(f"  Spearman Correlation: {corr:.4f} (p={p_value:.4e})")

    # Filter for plotting (log scale hates zeros)
    mask = (x_data > 0) & (y_data > 0)
    x_plot = x_data[mask]
    y_plot = y_data[mask]

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 7))
    if len(x_plot) > 0:
        hb = plt.hexbin(
            x_plot,
            y_plot,
            gridsize=30,
            cmap="inferno",
            bins="log",
            xscale="log",
            yscale="log",
        )
        plt.colorbar(hb, label="log10(Count)")
    else:
        plt.scatter(x_data, y_data, alpha=0.5)

    plt.xlabel("Citation Influence (PageRank)")
    plt.ylabel("Social Brokerage (Betweenness)")
    plt.title(f"Multiplex Correlation (r={corr:.2f})")
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(output_dir, "multiplex_centrality_correlation.png"))
    plt.close()

    return {"correlation": corr, "p_value": p_value}

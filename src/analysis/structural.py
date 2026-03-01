"""
Structural (graph-theoretic) analysis for the HEP-Th networks.

This module is intentionally kept free of any plotting code – it only computes
metrics and, in a few cases, writes small CSV helper tables that downstream
visualization utilities can consume.
"""

import logging
import os
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
from numba import jit
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


@jit(nopython=True)
def _fast_avg_strength(k_values: np.ndarray, s_values: np.ndarray, unique_k: np.ndarray) -> np.ndarray:
    """Average strength per unique degree (Numba for speed on large arrays)."""
    n_unique = len(unique_k)
    avg_s = np.zeros(n_unique, dtype=np.float64)
    for i in range(n_unique):
        k = unique_k[i]
        mask = k_values == k
        subset = s_values[mask]
        avg_s[i] = np.mean(subset)
    return avg_s


def _density_undirected(N: int, E: int) -> float:
    """Calculates the edge density for an undirected graph (like our co-authorship network)."""
    if N <= 1:
        return 0.0
    return float((2.0 * E) / (N * (N - 1)))


def _density_directed(N: int, E: int) -> float:
    """Calculates the edge density for a directed graph (like our citation network)."""
    if N <= 1:
        return 0.0
    return float(E / (N * (N - 1)))


def _directed_avg_clustering(G: nx.DiGraph) -> float:
    """
    Average directed local clustering coefficient per the report (Section 3.1.2).

    For each node i: C^dir_i = e^dir_i / (k^tot_i (k^tot_i - 1) - 2*k^↔_i),
    where e^dir_i is the number of directed triangles through i, k^tot_i = in_degree + out_degree,
    and k^↔_i is the number of reciprocal (bidirectional) edges incident to i.
    """
    if G.number_of_nodes() == 0:
        return 0.0
    total = 0.0
    for i in G:
        preds = set(G.predecessors(i))
        succs = set(G.successors(i))
        k_tot = len(preds) + len(succs)
        k_bidir = sum(1 for j in G.successors(i) if G.has_edge(j, i))
        if k_tot < 2:
            continue
        denom = k_tot * (k_tot - 1) - 2 * k_bidir
        if denom <= 0:
            continue
        e_dir = 0
        for u in preds:
            for v in succs:
                if u == v:
                    continue
                if G.has_edge(v, u) or G.has_edge(u, v):
                    e_dir += 1
        total += e_dir / denom
    return total / G.number_of_nodes()


def _directed_transitivity(G: nx.DiGraph) -> float:
    """
    Global directed transitivity: 3 * (number of directed triangles) / (number of open directed triads).

    Open directed triad = path of length 2 (u→v→w); closed = directed triangle (cycle of 3).
    """
    num_triangles = 0
    num_triples = 0
    for i in G:
        preds = list(G.predecessors(i))
        succs = list(G.successors(i))
        for u in preds:
            for v in succs:
                if u == v:
                    continue
                if G.has_edge(v, u) or G.has_edge(u, v):
                    num_triangles += 1
        in_d = len(preds)
        out_d = len(succs)
        k_bidir = sum(1 for j in succs if G.has_edge(j, i))
        num_triples += in_d * out_d - k_bidir
    num_triangles //= 3
    if num_triples <= 0:
        return 0.0
    return (3.0 * num_triangles) / num_triples


def get_global_metrics(G: nx.Graph) -> Dict[str, float]:
    """
    Grabs the big-picture metrics of the network. 
    This gives us a baseline understanding of how dense the collaborations are 
    and whether authors tend to form tight-knit triangles (clustering).
    """
    logger.info("--- Global Graph Metrics ---")
    n = G.number_of_nodes()
    e = G.number_of_edges()
    
    if G.is_directed():
        density = _density_directed(n, e)
        transitivity = _directed_transitivity(G)
        avg_clustering = _directed_avg_clustering(G)
    else:
        density = _density_undirected(n, e)
        transitivity = float(nx.transitivity(G))
        avg_clustering = float(nx.average_clustering(G))

    metrics = {
        "nodes": float(n),
        "edges": float(e),
        "density": float(density),
        "transitivity": float(transitivity),
        "avg_clustering": float(avg_clustering),
    }
    
    logger.info(f"Nodes: {int(metrics['nodes'])}, Edges: {int(metrics['edges'])}")
    logger.info(f"Edge Density: {metrics['density']:.6f}")
    logger.info(f"Global Clustering Coeff (Transitivity): {metrics['transitivity']:.4f}")
    logger.info(f"Average Clustering Coefficient: {metrics['avg_clustering']:.4f}")
    
    return metrics


def get_top_authors(G_co: nx.Graph, G_cit: nx.DiGraph) -> Dict[str, List[Tuple[str, float]]]:
    """
    Identifies the key players in our dataset across both layers.
    Finds the most collaborative authors (degree), the most cited authors (in-strength),
    and the "bridges" who connect different distinct research communities (betweenness).
    """
    logger.info("--- Centrality Analysis ---")

    top_connected = sorted(G_co.degree(), key=lambda x: x[1], reverse=True)[:5]
    logger.info("Most Collaborative (High Degree):")
    for author, degree in top_connected:
        logger.info(f"  - {author}: {degree} co-authors")

    top_cited = sorted(G_cit.in_degree(weight="weight"), key=lambda x: x[1], reverse=True)[:5]
    logger.info("Most Influential (Highest citation in-strength):")
    for author, count in top_cited:
        logger.info(f"  - {author}: {count:.1f} total incoming citations")

    logger.info("Calculating Betweenness Centrality (this may take a moment)...")
    top_bridges: List[Tuple[str, float]] = []
    try:
        # We sample up to 500 nodes to keep computation time reasonable.
        k_sample = min(500, max(1, G_co.number_of_nodes() - 1))
        betweenness = nx.betweenness_centrality(G_co, weight="distance", k=k_sample)
        top_bridges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("Top Bridges (High Betweenness):")
        for author, score in top_bridges:
            logger.info(f"  - {author}: {score:.4f}")
    except (KeyError, ValueError) as e:
        logger.error(f"Error calculating betweenness: {e}")

    logger.info("Calculating Closeness Centrality...")
    top_closeness: List[Tuple[str, float]] = []
    try:
        closeness = nx.closeness_centrality(G_co, distance="distance")
        top_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("Top Closeness:")
        for author, score in top_closeness:
            logger.info(f"  - {author}: {score:.4f}")
    except (KeyError, ValueError) as e:
        logger.error(f"Error calculating closeness: {e}")

    return {
        "collaborative": [(a, float(v)) for a, v in top_connected],
        "influential": [(a, float(v)) for a, v in top_cited],
        "bridges": [(a, float(v)) for a, v in top_bridges],
        "closeness": [(a, float(v)) for a, v in top_closeness],
    }


def compute_centrality_data(G_co: nx.Graph, top_n: int = 10) -> Dict[str, Any]:
    """
    Pure centrality computation for the social layer.

    This function performs no file I/O and is therefore easy to test in
    isolation. It returns the top-N betweenness and closeness pairs which can
    then be written to disk or plotted by downstream utilities.
    """
    k_sample = min(500, max(1, G_co.number_of_nodes() - 1))
    betweenness = nx.betweenness_centrality(G_co, weight="distance", k=k_sample)
    closeness = nx.closeness_centrality(G_co, distance="distance")

    top_bet = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_clo = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return {
        "top_betweenness_data": top_bet,
        "top_closeness_data": top_clo,
    }


def export_centrality_tables(G_co: nx.Graph, output_dir: str = "results", top_n: int = 10) -> Dict[str, Any]:
    """
    Computes betweenness and closeness centralities, saves the top N to CSV files,
    and returns the raw data so our visualization layer can plot them later.
    """
    os.makedirs(output_dir, exist_ok=True)

    centrality_data = compute_centrality_data(G_co, top_n=top_n)
    top_bet = centrality_data["top_betweenness_data"]
    top_clo = centrality_data["top_closeness_data"]

    bet_path = os.path.join(output_dir, "top_betweenness_social.csv")
    clo_path = os.path.join(output_dir, "top_closeness_social.csv")

    with open(bet_path, "w", encoding="utf-8") as f:
        f.write("author,betweenness\n")
        for a, v in top_bet:
            f.write(f"{a},{v:.8f}\n")

    with open(clo_path, "w", encoding="utf-8") as f:
        f.write("author,closeness\n")
        for a, v in top_clo:
            f.write(f"{a},{v:.8f}\n")

    logger.info(f"Saved table to: {bet_path}")
    logger.info(f"Saved table to: {clo_path}")

    return {
        "betweenness_csv": bet_path,
        "closeness_csv": clo_path,
        "top_betweenness_data": top_bet,
        "top_closeness_data": top_clo,
    }


def analyze_layer_shortest_paths(G_cit: nx.DiGraph, G_co: nx.Graph) -> Dict[str, Any]:
    """
    Investigates how the citation layer maps onto the social layer. 
    If author A cites author B, how many steps away are they in the co-authorship network?
    Returns the average distances along with the raw distributions for plotting.
    """
    logger.info("--- Cross-Layer Path Analysis ---")

    social_nodes = set(G_co.nodes())
    valid_edges = [(u, v) for u, v in G_cit.edges() if u in social_nodes and v in social_nodes]

    distances_weighted: List[float] = []
    distances_hops: List[int] = []

    for u, v in valid_edges:
        try:
            d_hops = nx.shortest_path_length(G_co, source=u, target=v)
            if np.isfinite(d_hops) and d_hops > 0:
                distances_hops.append(int(d_hops))
        except nx.NetworkXNoPath:
            pass

        try:
            d_w = nx.shortest_path_length(G_co, source=u, target=v, weight="distance")
            if np.isfinite(d_w) and d_w > 0:
                distances_weighted.append(float(d_w))
        except nx.NetworkXNoPath:
            pass

    avg_hops = float(np.mean(distances_hops)) if distances_hops else 0.0
    avg_weighted = float(np.mean(distances_weighted)) if distances_weighted else 0.0

    logger.info(f"Analyzed {len(valid_edges)} citation edges on shared node set.")
    logger.info(f"Average Co-authorship distance (hops): {avg_hops:.4f}")
    logger.info(f"Average Co-authorship distance (weighted): {avg_weighted:.4f}")

    return {
        "avg_hops": float(avg_hops),
        "avg_weighted": float(avg_weighted),
        "distances_hops": distances_hops,
        "distances_weighted": distances_weighted,
    }


def analyze_strength_distribution(G: nx.Graph, name: str = "Network") -> Dict[str, Any]:
    """
    Analyzes the correlation between an author's degree (number of connections) 
    and their strength (total weight of those connections).
    Returns the fitting parameters (beta, intercept) and the raw arrays for visualization.
    """
    logger.info(f"--- Weighted Strength Analysis ({name}) ---")

    if G.is_directed():
        degrees = dict(G.in_degree())
        strengths = dict(G.in_degree(weight="weight"))
    else:
        degrees = dict(G.degree())
        strengths = dict(G.degree(weight="weight"))

    nodes = list(G.nodes())
    k_values = np.array([degrees.get(n, 0) for n in nodes], dtype=np.float64)
    s_values = np.array([strengths.get(n, 0.0) for n in nodes], dtype=np.float64)

    mask = (k_values > 0) & (s_values > 0)
    k_values = k_values[mask]
    s_values = s_values[mask]

    if len(k_values) == 0:
        logger.warning("No positive degree/strength values found.")
        return {"beta": 0.0, "intercept": 0.0, "k_values": [], "s_values": []}

    k_unique = np.unique(k_values)
    s_avg_k = _fast_avg_strength(k_values, s_values, k_unique)

    log_k = np.log10(k_unique)
    log_s = np.log10(s_avg_k)

    if len(log_k) > 1:
        beta, intercept = np.polyfit(log_k, log_s, 1)
    else:
        beta, intercept = 0.0, 0.0

    logger.info(f"  Fit exponent (beta): {beta:.4f}")
    if beta > 1.0:
        logger.info("  -> Super-linear scaling (beta > 1).")

    return {
        "beta": float(beta),
        "intercept": float(intercept),
        "k_values": k_values,
        "s_values": s_values,
        "k_unique": k_unique,
        "s_avg_k": s_avg_k,
    }


def analyze_multiplex_correlation(G_co: nx.Graph, G_cit: nx.DiGraph) -> Dict[str, Any]:
    """
    Compares how influential an author is in the citation network (PageRank)
    versus their bridging capability in the social network (Betweenness).
    Returns the Spearman correlation and the raw arrays for generating hexbin plots.
    """
    logger.info("--- Multiplex Correlation Analysis ---")

    common_authors = list(set(G_co.nodes()).intersection(set(G_cit.nodes())))
    if len(common_authors) < 10:
        logger.warning("Not enough common authors for correlation.")
        return {"correlation": 0.0, "p_value": 1.0, "x_plot": [], "y_plot": []}

    pagerank = nx.pagerank(G_cit, weight="weight")
    k_sample = min(len(common_authors), 500)
    betweenness = nx.betweenness_centrality(G_co, weight="distance", k=k_sample)

    x_data = np.array([pagerank.get(a, 0.0) for a in common_authors], dtype=float)
    y_data = np.array([betweenness.get(a, 0.0) for a in common_authors], dtype=float)

    corr, p_value = spearmanr(x_data, y_data)
    corr = float(corr) if np.isfinite(corr) else 0.0
    p_value = float(p_value) if np.isfinite(p_value) else 1.0

    logger.info(f"  Spearman Correlation: {corr:.4f} (p={p_value:.4e})")

    mask = (x_data > 0) & (y_data > 0)
    
    return {
        "correlation": corr, 
        "p_value": p_value,
        "x_plot": x_data[mask],
        "y_plot": y_data[mask],
        "x_raw": x_data,
        "y_raw": y_data
    }


def analyze_degree_correlation(G_co: nx.Graph, G_cit: nx.DiGraph) -> Dict[str, Any]:
    """
    Analyzes the relationship between an author's number of collaborators (degree)
    and their total incoming citations (in-strength). Identifies notable outliers.
    Returns the Spearman correlation and the raw arrays for generating hexbin plots.
    """
    logger.info("--- Degree vs In-Strength Correlation ---")

    common_authors = list(set(G_co.nodes()).intersection(set(G_cit.nodes())))
    if len(common_authors) < 10:
        return {"correlation": 0.0, "p_value": 1.0, "x_plot": [], "y_plot": []}

    co_degrees = dict(G_co.degree())
    cit_in_strength = dict(G_cit.in_degree(weight="weight"))

    x_data = np.array([co_degrees.get(a, 0) for a in common_authors], dtype=float)
    y_data = np.array([cit_in_strength.get(a, 0.0) for a in common_authors], dtype=float)

    corr, p_value = spearmanr(x_data, y_data)
    corr = float(corr) if np.isfinite(corr) else 0.0
    p_value = float(p_value) if np.isfinite(p_value) else 1.0
    logger.info(f"  Spearman Correlation: {corr:.4f} (p={p_value:.4e})")

    outliers_cit = sorted(
        [a for a in common_authors if co_degrees.get(a, 0) > 0],
        key=lambda a: (cit_in_strength.get(a, 0.0) / co_degrees.get(a, 0)),
        reverse=True,
    )[:5]
    logger.info("High citations (in-strength) / Low collaboration (Outliers):")
    for a in outliers_cit:
        logger.info(f"  - {a}: {cit_in_strength.get(a, 0.0):.1f} citations, {co_degrees.get(a, 0)} co-authors")

    outliers_co = sorted(
        [a for a in common_authors],
        key=lambda a: (co_degrees.get(a, 0) / (cit_in_strength.get(a, 0.0) + 1.0)),
        reverse=True,
    )[:5]
    logger.info("High collaboration / Low citations (in-strength) (Outliers):")
    for a in outliers_co:
        logger.info(f"  - {a}: {co_degrees.get(a, 0)} co-authors, {cit_in_strength.get(a, 0.0):.1f} citations")

    mask = (x_data > 0) & (y_data > 0)

    return {
        "correlation": corr, 
        "p_value": p_value,
        "x_plot": x_data[mask],
        "y_plot": y_data[mask],
        "x_raw": x_data,
        "y_raw": y_data
    }
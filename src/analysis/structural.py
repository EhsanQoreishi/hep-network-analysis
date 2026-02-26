import logging
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numba import jit
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

@jit(nopython=True)
def _fast_avg_strength(k_values: np.ndarray, s_values: np.ndarray, unique_k: np.ndarray) -> np.ndarray:
    n_unique = len(unique_k)
    avg_s = np.zeros(n_unique, dtype=np.float64)
    for i in range(n_unique):
        k = unique_k[i]
        mask = k_values == k
        subset = s_values[mask]
        avg_s[i] = np.mean(subset)
    return avg_s

def _sanitize_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("\\\\", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(":", "")
        .replace(",", "")
        .replace("__", "_")
    )

def _savefig(path: str) -> None:
    base, ext = os.path.splitext(path)
    if ext.lower() != ".pdf":
        path = base + ".pdf"
    plt.savefig(path, format="pdf", bbox_inches="tight")
    logger.info(f"Saved figure to: {path}")

def _density_undirected(N: int, E: int) -> float:
    if N <= 1:
        return 0.0
    return float((2.0 * E) / (N * (N - 1)))

def _density_directed(N: int, E: int) -> float:
    if N <= 1:
        return 0.0
    return float(E / (N * (N - 1)))

def get_global_metrics(G: nx.Graph) -> Dict[str, float]:
    logger.info("--- Global Graph Metrics ---")
    n = G.number_of_nodes()
    e = G.number_of_edges()
    if G.is_directed():
        density = _density_directed(n, e)
    else:
        density = _density_undirected(n, e)
    metrics = {
        "nodes": float(n),
        "edges": float(e),
        "density": float(density),
        "transitivity": float(nx.transitivity(G)),
        "avg_clustering": float(nx.average_clustering(G)),
    }
    logger.info(f"Nodes: {int(metrics['nodes'])}, Edges: {int(metrics['edges'])}")
    logger.info(f"Edge Density: {metrics['density']:.6f}")
    logger.info(f"Global Clustering Coeff (Transitivity): {metrics['transitivity']:.4f}")
    logger.info(f"Average Clustering Coefficient: {metrics['avg_clustering']:.4f}")
    return metrics

def get_top_authors(G_co: nx.Graph, G_cit: nx.DiGraph) -> Dict[str, List[Tuple[str, float]]]:
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

def export_centrality_tables(G_co: nx.Graph, output_dir: str = "results", top_n: int = 10) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    k_sample = min(500, max(1, G_co.number_of_nodes() - 1))
    betweenness = nx.betweenness_centrality(G_co, weight="distance", k=k_sample)
    closeness = nx.closeness_centrality(G_co, distance="distance")

    top_bet = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_clo = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:top_n]

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

    return {"betweenness_csv": bet_path, "closeness_csv": clo_path}

def plot_top_centralities(
    G_co: nx.Graph,
    output_dir: str = "results",
    top_n: int = 10,
    filename: str = "centrality_betweenness_closeness_social.pdf",
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    k_sample = min(500, max(1, G_co.number_of_nodes() - 1))
    betweenness = nx.betweenness_centrality(G_co, weight="distance", k=k_sample)
    closeness = nx.closeness_centrality(G_co, distance="distance")

    top_bet = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_clo = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:top_n]

    bet_names = [a for a, _ in top_bet][::-1]
    bet_vals = [v for _, v in top_bet][::-1]
    clo_names = [a for a, _ in top_clo][::-1]
    clo_vals = [v for _, v in top_clo][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].barh(bet_names, bet_vals, color="#4C72B0", edgecolor="black", linewidth=0.5)
    axes[0].set_title(f"Top {top_n} Betweenness (Social Layer)")
    axes[0].set_xlabel("Betweenness")
    axes[0].grid(True, axis="x", alpha=0.25)

    axes[1].barh(clo_names, clo_vals, color="#55A868", edgecolor="black", linewidth=0.5)
    axes[1].set_title(f"Top {top_n} Closeness (Social Layer)")
    axes[1].set_xlabel("Closeness")
    axes[1].grid(True, axis="x", alpha=0.25)

    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    _savefig(save_path)
    plt.close(fig)
    return os.path.splitext(save_path)[0] + ".pdf"

def analyze_layer_shortest_paths(
    G_cit: nx.DiGraph,
    G_co: nx.Graph,
    output_dir: str = "results",
    filename: str = "cross_layer_path_distribution.pdf",
) -> Dict[str, float]:
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

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(9, 5))
    if distances_hops:
        min_d = int(min(distances_hops))
        max_d = int(max(distances_hops))
        bins = np.arange(min_d - 0.5, max_d + 1.5, 1.0)
        plt.hist(distances_hops, bins=bins, alpha=0.75, color="skyblue", edgecolor="black")
        plt.xticks(list(range(min_d, max_d + 1)))
    else:
        plt.hist([], bins=10)
    plt.title("Cross-layer Path Distribution (Citation pairs in Social layer)")
    plt.xlabel("Shortest path length d (co-authorship hops)")
    plt.ylabel("Frequency")
    if distances_hops:
        plt.axvline(avg_hops, color="red", linestyle="dashed", label=f"Mean: {avg_hops:.4f}")
        plt.legend()
    save_path = os.path.join(output_dir, "cross_layer_path_distribution_hops.pdf")
    _savefig(save_path)
    plt.close()

    plt.figure(figsize=(9, 5))
    if distances_weighted:
        plt.hist(distances_weighted, bins=50, alpha=0.75, color="skyblue", edgecolor="black")
    else:
        plt.hist([], bins=10)
    plt.title("Cross-layer Path Distribution (Citation pairs in Social layer)")
    plt.xlabel("Shortest path length d (weighted social distance)")
    plt.ylabel("Frequency")
    if distances_weighted:
        plt.axvline(avg_weighted, color="red", linestyle="dashed", label=f"Mean: {avg_weighted:.4f}")
        plt.legend()
    save_path = os.path.join(output_dir, "cross_layer_path_distribution_weighted.pdf")
    _savefig(save_path)
    plt.close()

    return {"avg_hops": float(avg_hops), "avg_weighted": float(avg_weighted)}

def analyze_strength_distribution(G: nx.Graph, name: str = "Network", output_dir: str = "results") -> Dict[str, float]:
    logger.info(f"--- Weighted Strength Analysis ({name}) ---")

    if G.is_directed():
        degrees = dict(G.in_degree())
        strengths = dict(G.in_degree(weight="weight"))
        degree_label = "In-Degree k"
        strength_label = "In-Strength s"
    else:
        degrees = dict(G.degree())
        strengths = dict(G.degree(weight="weight"))
        degree_label = "Degree k"
        strength_label = "Strength s"

    nodes = list(G.nodes())
    k_values = np.array([degrees.get(n, 0) for n in nodes], dtype=np.float64)
    s_values = np.array([strengths.get(n, 0.0) for n in nodes], dtype=np.float64)

    mask = (k_values > 0) & (s_values > 0)
    k_values = k_values[mask]
    s_values = s_values[mask]

    if len(k_values) == 0:
        logger.warning("No positive degree/strength values found.")
        return {"beta": 0.0, "intercept": 0.0}

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

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8.5, 6))
    plt.scatter(k_values, s_values, alpha=0.12, color="gray", s=10, label="Nodes")
    plt.loglog(k_unique, s_avg_k, "o", color="#1f77b4", label=r"Average $\langle s(k)\rangle$")

    fit_y = (10 ** intercept) * (k_unique ** beta)
    plt.loglog(k_unique, fit_y, "--", color="red", linewidth=2, label=fr"Fit: $\beta={beta:.2f}$")

    plt.title(f"Strength vs Degree ({name})")
    plt.xlabel(degree_label)
    plt.ylabel(strength_label)
    plt.legend()
    plt.grid(True, which="both", alpha=0.25)

    safe_name = _sanitize_name(name)
    save_path = os.path.join(output_dir, f"{safe_name}_strength_degree_correlation.pdf")
    _savefig(save_path)
    plt.close()

    return {"beta": float(beta), "intercept": float(intercept)}

def analyze_multiplex_correlation(G_co: nx.Graph, G_cit: nx.DiGraph, output_dir: str = "results") -> Dict[str, float]:
    logger.info("--- Multiplex Correlation Analysis ---")

    common_authors = list(set(G_co.nodes()).intersection(set(G_cit.nodes())))
    if len(common_authors) < 10:
        logger.warning("Not enough common authors for correlation.")
        return {"correlation": 0.0, "p_value": 1.0}

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

    plt.xlabel("Citation influence (PageRank)")
    plt.ylabel("Social brokerage (Betweenness)")
    plt.title(f"Multiplex Correlation (Spearman r={corr:.2f})")
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, "multiplex_pagerank_vs_betweenness.pdf")
    _savefig(save_path)
    plt.close()

    return {"correlation": corr, "p_value": p_value}

def analyze_degree_correlation(G_co: nx.Graph, G_cit: nx.DiGraph, output_dir: str = "results") -> Dict[str, float]:
    logger.info("--- Degree vs In-Strength Correlation ---")

    common_authors = list(set(G_co.nodes()).intersection(set(G_cit.nodes())))
    if len(common_authors) < 10:
        return {"correlation": 0.0, "p_value": 1.0}

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
        logger.info(
            f"  - {a}: {cit_in_strength.get(a, 0.0):.1f} citations, {co_degrees.get(a, 0)} co-authors"
        )

    outliers_co = sorted(
        [a for a in common_authors],
        key=lambda a: (co_degrees.get(a, 0) / (cit_in_strength.get(a, 0.0) + 1.0)),
        reverse=True,
    )[:5]
    logger.info("High collaboration / Low citations (in-strength) (Outliers):")
    for a in outliers_co:
        logger.info(
            f"  - {a}: {co_degrees.get(a, 0)} co-authors, {cit_in_strength.get(a, 0.0):.1f} citations"
        )

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 7))

    mask = (x_data > 0) & (y_data > 0)
    x_plot = x_data[mask]
    y_plot = y_data[mask]

    if len(x_plot) > 0:
        hb = plt.hexbin(
            x_plot,
            y_plot,
            gridsize=30,
            cmap="viridis",
            bins="log",
            xscale="log",
            yscale="log",
        )
        plt.colorbar(hb, label="log10(Count)")
    else:
        plt.scatter(x_data, y_data, alpha=0.5)

    plt.xlabel("Co-authorship degree")
    plt.ylabel("Citation in-strength (total citations received)")
    plt.title(f"Degree vs In-Strength (Spearman r={corr:.2f})")
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, "degree_vs_instrength_social_vs_citation.pdf")
    _savefig(save_path)
    plt.close()

    return {"correlation": corr, "p_value": p_value}

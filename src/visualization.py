"""
Visualization and figure/HTML generation for the HEP-Th analysis.

All numeric analysis and physics logic lives in ``src.analysis.*``. This module
assumes those functions have already computed the required data structures and
focuses purely on turning them into static plots (PDF) or interactive
visualizations (HTML via pyvis).
"""

import logging
import os
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyvis.network import Network
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)


def _sanitize_name(name: str) -> str:
    """
    Cleans up layer names or network names so they can be safely used as file names.
    """
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(":", "")
        .replace(",", "")
        .replace("__", "_")
    )


def _savefig(path: str) -> None:
    """Save current figure as PDF (adds .pdf if missing)."""
    base, ext = os.path.splitext(path)
    if ext.lower() != ".pdf":
        path = base + ".pdf"
    plt.savefig(path, format="pdf", bbox_inches="tight")
    logger.info(f"Saved figure to: {path}")


# =============================================================================
# INTERACTIVE NETWORK TOPOLOGY
# =============================================================================

def visualize_network(
    G: nx.Graph,
    title: str = "results/hep_interactive_map.html",
    partition: Dict[Any, int] = None,
    degree_full: Dict[Any, int] = None,
) -> None:
    """
    Build an interactive PyVis map of the graph, with nodes colored by community.

    The community partition must be computed by the analysis layer (e.g. via
    get_community_partition) and passed in; this module does not run Louvain.

    If degree_full is provided (e.g. from the full social graph), tooltips and
    node sizes use full-network degree so the map shows global stats while
    rendering a subgraph.
    """
    logger.info("--- Projecting Interactive Topological Map ---")
    
    # Check if graph is empty before proceeding
    if G.number_of_nodes() == 0:
        logger.warning("Attempted to visualize an empty graph. Skipping.")
        return

    if partition is None:
        logger.warning("No partition provided; all nodes will use community id 0.")
        partition = {}

    output_dir = os.path.dirname(title)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    degrees_subgraph = dict(G.degree())
    # Use full-network degree for display when provided; else subgraph degree
    degrees_display = degree_full if degree_full is not None else degrees_subgraph

    logger.info(f"  Rendering subgraph with {G.number_of_nodes()} nodes...")

    net = Network(height="100vh", width="100%", bgcolor="#222222", font_color="white")

    for node in G.nodes():
        comm_id = partition.get(node, 0)
        degree = degrees_display.get(node, degrees_subgraph.get(node, 0))
        net.add_node(
            node,
            label=node,
            title=f"Author: {node}\nDegree: {degree}\nCommunity: {comm_id}",
            value=degree,
            group=comm_id,
        )

    for u, v in G.edges():
        net.add_edge(u, v, color="#555555", alpha=0.3)

    net.force_atlas_2based()

    try:
        net.save_graph(title)
        logger.info(f"  Success! Interactive map saved to: {title}")
    except Exception as e:
        logger.error(f"  Error saving visualization: {e}")


# =============================================================================
# STRUCTURAL PLOTS
# =============================================================================

def plot_top_centralities(data: Dict[str, Any], output_dir: str = "results", top_n: int = 10) -> None:
    """Plots horizontal bar charts showing the authors with the highest betweenness and closeness."""
    top_bet = data.get("top_betweenness_data", [])
    top_clo = data.get("top_closeness_data", [])
    
    # Exit before directory creation if data is empty
    if not top_bet or not top_clo:
        return

    os.makedirs(output_dir, exist_ok=True)
    
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
    _savefig(os.path.join(output_dir, "centrality_betweenness_closeness_social.pdf"))
    plt.close(fig)


def plot_cross_layer_paths(data: Dict[str, Any], output_dir: str = "results") -> None:
    """Visualizes how far apart citing authors are in the co-authorship layer."""
    distances_hops = data.get("distances_hops", [])
    distances_weighted = data.get("distances_weighted", [])
    
    if not distances_hops and not distances_weighted:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    avg_hops = data.get("avg_hops", 0.0)
    avg_weighted = data.get("avg_weighted", 0.0)

    # Plot Hops
    plt.figure(figsize=(9, 5))
    if distances_hops:
        min_d = int(min(distances_hops))
        max_d = int(max(distances_hops))
        bins = np.arange(min_d - 0.5, max_d + 1.5, 1.0)
        plt.hist(distances_hops, bins=bins, alpha=0.75, color="skyblue", edgecolor="black")
        plt.xticks(list(range(min_d, max_d + 1)))
        plt.axvline(avg_hops, color="red", linestyle="dashed", label=f"Mean: {avg_hops:.4f}")
        plt.legend()
    else:
        plt.hist([], bins=10)
    plt.title("Cross-layer Path Distribution (Citation pairs in Social layer)")
    plt.xlabel("Shortest path length d (co-authorship hops)")
    plt.ylabel("Frequency")
    _savefig(os.path.join(output_dir, "cross_layer_path_distribution_hops.pdf"))
    plt.close()

    # Plot Weighted
    plt.figure(figsize=(9, 5))
    if distances_weighted:
        plt.hist(distances_weighted, bins=50, alpha=0.75, color="skyblue", edgecolor="black")
        plt.axvline(avg_weighted, color="red", linestyle="dashed", label=f"Mean: {avg_weighted:.4f}")
        plt.legend()
    else:
        plt.hist([], bins=10)
    plt.title("Cross-layer Path Distribution (Citation pairs in Social layer)")
    plt.xlabel("Shortest path length d (weighted social distance)")
    plt.ylabel("Frequency")
    _savefig(os.path.join(output_dir, "cross_layer_path_distribution_weighted.pdf"))
    plt.close()


def plot_strength_distribution(data: Dict[str, Any], name: str = "Network", output_dir: str = "results") -> None:
    """Plots the correlation between an author's degree and their connection strength."""
    k_values = data.get("k_values", [])
    if len(k_values) == 0:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    s_values = data.get("s_values", [])
    k_unique = data.get("k_unique", [])
    s_avg_k = data.get("s_avg_k", [])
    beta = data.get("beta", 0.0)
    intercept = data.get("intercept", 0.0)

    plt.figure(figsize=(8.5, 6))
    plt.scatter(k_values, s_values, alpha=0.12, color="gray", s=10, label="Nodes")
    plt.loglog(k_unique, s_avg_k, "o", color="#1f77b4", label=r"Average $\langle s(k)\rangle$")

    fit_y = (10 ** intercept) * (k_unique ** beta)
    plt.loglog(k_unique, fit_y, "--", color="red", linewidth=2, label=fr"Fit: $\beta={beta:.2f}$")

    plt.title(f"Strength vs Degree ({name})")
    plt.xlabel("Degree k" if "Social" in name else "In-Degree k")
    plt.ylabel("Strength s" if "Social" in name else "In-Strength s")
    plt.legend()
    plt.grid(True, which="both", alpha=0.25)

    _savefig(os.path.join(output_dir, f"{_sanitize_name(name)}_strength_degree_correlation.pdf"))
    plt.close()


def plot_multiplex_correlation(data: Dict[str, Any], output_dir: str = "results") -> None:
    """Generates a hexbin plot comparing Citation PageRank vs Social Betweenness."""
    x_plot = data.get("x_plot", [])
    y_plot = data.get("y_plot", [])
    
    if len(x_plot) == 0:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    corr = data.get("correlation", 0.0)

    plt.figure(figsize=(10, 7))
    hb = plt.hexbin(
        x_plot, y_plot, gridsize=30, cmap="inferno", bins="log", xscale="log", yscale="log"
    )
    plt.colorbar(hb, label="log10(Count)")

    plt.xlabel("Citation influence (PageRank)")
    plt.ylabel("Social brokerage (Betweenness)")
    plt.title(f"Multiplex Correlation (Spearman r={corr:.2f})")
    plt.grid(True, alpha=0.3)

    _savefig(os.path.join(output_dir, "multiplex_pagerank_vs_betweenness.pdf"))
    plt.close()


def plot_degree_vs_instrength(data: Dict[str, Any], output_dir: str = "results") -> None:
    """Generates a hexbin plot comparing Co-authorship degree vs total incoming citations."""
    x_plot = data.get("x_plot", [])
    y_plot = data.get("y_plot", [])
    
    if len(x_plot) == 0:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    corr = data.get("correlation", 0.0)

    plt.figure(figsize=(10, 7))
    hb = plt.hexbin(
        x_plot, y_plot, gridsize=30, cmap="viridis", bins="log", xscale="log", yscale="log"
    )
    plt.colorbar(hb, label="log10(Count)")

    plt.xlabel("Co-authorship degree")
    plt.ylabel("Citation in-strength (total citations received)")
    plt.title(f"Degree vs In-Strength (Spearman r={corr:.2f})")
    plt.grid(True, alpha=0.3)

    _savefig(os.path.join(output_dir, "degree_vs_instrength_social_vs_citation.pdf"))
    plt.close()


# =============================================================================
# PHYSICS PLOTS
# =============================================================================

def plot_power_law(data: Dict[str, Any], name: str = "Network", output_dir: str = "results") -> None:
    """
    Visualizes the heavy-tailed degree distribution and its power-law fit.

    Uses the powerlaw library's plot_pdf (same as the old version) when
    analyze_power_law provides a "fit" object; otherwise falls back to
    precomputed PDF curves (pdf_x, pdf_empirical, etc.).
    """
    fit = data.get("fit")
    if fit is not None:
        # Same as old version: use powerlaw's built-in plotting
        os.makedirs(output_dir, exist_ok=True)
        try:
            plt.figure(figsize=(8, 6))
            fit.plot_pdf(color="b", linear_bins=True, label="Empirical Data")
            fit.power_law.plot_pdf(color="r", linestyle="--", label="Power Law Fit")
            fit.lognormal.plot_pdf(color="g", linestyle="-.", label="Log-Normal Fit")
            plt.title(f"Degree Distribution ({name})")
            plt.xlabel(data.get("x_label", "Degree (k)"))
            plt.ylabel("P(k)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            _savefig(os.path.join(output_dir, f"{_sanitize_name(name)}_degree_distribution_powerlaw.pdf"))
            plt.close()
        except Exception as e:
            logger.error(f"Failed to plot power law for {name}: {e}")
        return

    # Fallback: precomputed arrays (e.g. tests or when fit not provided)
    x_plot = data.get("pdf_x")
    if x_plot is None or len(x_plot) < 2:
        return

    os.makedirs(output_dir, exist_ok=True)
    pdf_empirical = data.get("pdf_empirical", np.array([]))
    pdf_power_law = data.get("pdf_power_law", np.array([]))
    pdf_lognormal = data.get("pdf_lognormal", np.array([]))
    x_pl = data.get("pdf_power_law_x")
    x_ln = data.get("pdf_lognormal_x")

    try:
        plt.figure(figsize=(8, 6))
        plt.loglog(x_plot, pdf_empirical, "b", label="Empirical Data")
        if len(pdf_power_law) > 0 and np.any(np.isfinite(pdf_power_law)):
            x_pl_use = x_pl if x_pl is not None and len(x_pl) == len(pdf_power_law) else (x_plot if len(x_plot) == len(pdf_power_law) else None)
            if x_pl_use is not None:
                plt.loglog(x_pl_use, pdf_power_law, "r--", linewidth=2, label="Power Law Fit")
        if len(pdf_lognormal) > 0 and np.any(np.isfinite(pdf_lognormal)):
            x_ln_use = x_ln if x_ln is not None and len(x_ln) == len(pdf_lognormal) else (x_plot if len(x_plot) == len(pdf_lognormal) else None)
            if x_ln_use is not None:
                plt.loglog(x_ln_use, pdf_lognormal, "g-.", linewidth=1.5, label="Log-Normal Fit")
        plt.title(f"Degree Distribution ({name})")
        plt.xlabel(data.get("x_label", "Degree (k)"))
        plt.ylabel("P(k)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        _savefig(os.path.join(output_dir, f"{_sanitize_name(name)}_degree_distribution_powerlaw.pdf"))
        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot power law for {name}: {e}")


def plot_spectral_density(data: Dict[str, Any], name: str = "Network", output_dir: str = "results") -> None:
    """Plots the distribution of the Laplacian eigenvalues to visualize spectral density."""
    eigenvalues = data.get("eigenvalues")
    if eigenvalues is None or len(eigenvalues) == 0:
        return
    
    os.makedirs(output_dir, exist_ok=True)
        
    vn_entropy = data.get("vn_entropy", 0.0)

    plt.figure(figsize=(10, 6))
    try:
        kde = gaussian_kde(eigenvalues)
        x_range = np.linspace(float(np.min(eigenvalues)), float(np.max(eigenvalues)), 200)
        y_kde = kde(x_range)
        plt.plot(x_range, y_kde, color="blue", lw=2, label="Spectral Density")
        plt.fill_between(x_range, y_kde, color="blue", alpha=0.1)
        plt.legend()
    except Exception:
        plt.hist(eigenvalues, bins=50, color="blue", alpha=0.5)

    plt.title(f"Spectral Density ({name}) (Entropy S={vn_entropy:.2f})")
    plt.xlabel(r"Eigenvalue ($\lambda$)")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)

    _savefig(os.path.join(output_dir, f"{_sanitize_name(name)}_spectral_density_entropy.pdf"))
    plt.close()


def plot_robustness(data: Dict[str, Any], name: str = "Network", output_dir: str = "results") -> None:
    """Compares the network's resilience under random failure vs targeted hub attacks."""
    x_axis = data.get("x_axis", [])
    random_sizes = data.get("random_sizes", [])
    attack_sizes = data.get("attack_sizes", [])
    
    if not x_axis:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8.5, 6))
    plt.plot(x_axis, random_sizes, "g-o", label="Random Failure")
    plt.plot(x_axis, attack_sizes, "r-s", label="Targeted Attack (Degree hubs)")
    plt.title(f"Network Robustness ({name})")
    plt.xlabel("Fraction of nodes removed (f)")
    plt.ylabel(r"Normalized GCC size $s(f) = S(f)/N$")
    plt.legend()
    plt.grid(True, alpha=0.3)

    _savefig(os.path.join(output_dir, f"{_sanitize_name(name)}_network_robustness.pdf"))
    plt.close()


def plot_configuration_model(data: Dict[str, Any], name: str = "Network", output_dir: str = "results") -> None:
    """Histograms the null model clustering values and marks the real network's clustering."""
    null_clustering_values = data.get("null_clustering_values", [])
    if not null_clustering_values:
        return

    os.makedirs(output_dir, exist_ok=True)

    C_real = data.get("C_real", 0.0)
    z_score = data.get("z_score", 0.0)

    plt.figure(figsize=(8, 5))
    plt.hist(null_clustering_values, color="gray", alpha=0.7, label="Null Model")
    plt.axvline(C_real, color="red", linestyle="dashed", linewidth=2, label="Real Network")
    plt.title(f"Configuration-Model Clustering ({name}) (Z={z_score:.2f})")
    plt.xlabel("Average clustering coefficient")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.25)

    _savefig(os.path.join(output_dir, f"{_sanitize_name(name)}_configuration_model_clustering.pdf"))
    plt.close()


# =============================================================================
# COMMUNITIES PLOTS
# =============================================================================

def plot_community_distribution(data: Dict[str, Any], layer_name: str = "Network", output_dir: str = "results") -> None:
    """Plots a histogram of the sizes of the detected communities in the network."""
    sizes = data.get("sizes", [])
    # Exit before directory creation if data is empty
    if not sizes:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    plt.hist(sizes, bins=50, color="teal", edgecolor="black")
    plt.title(f"Community Size Distribution ({layer_name})")
    plt.xlabel("Community size (number of authors)")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.grid(axis="y", alpha=0.5)

    _savefig(os.path.join(output_dir, f"{_sanitize_name(layer_name)}_community_size_distribution.pdf"))
    plt.close()
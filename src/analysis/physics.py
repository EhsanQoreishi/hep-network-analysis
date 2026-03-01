import logging
import random
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import powerlaw
import scipy.linalg
import scipy.sparse.linalg

logger = logging.getLogger(__name__)


def _ensure_distance_attribute(G: nx.Graph) -> None:
    """
    Utility function to ensure all edges have a 'distance' attribute.
    Since weight often means 'number of collaborations', a higher weight 
    means authors are closer. Distance is calculated as 1 / weight.
    """
    for u, v, d in G.edges(data=True):
        if "distance" in d:
            continue
        w = d.get("weight", None)
        if w is None:
            d["distance"] = 1.0
        else:
            try:
                wf = float(w)
                d["distance"] = (1.0 / wf) if wf > 0 else 1.0
            except Exception:
                d["distance"] = 1.0


def analyze_power_law(G: nx.Graph, name: str = "Network") -> Dict[str, Any]:
    """
    Checks if the network degree distribution follows a heavy-tailed power law.
    
    Returns the fit parameters and the raw degree array for plotting.
    Handles mathematical edge cases (like Star graphs) where fits may fail.
    """
    logger.info(f"--- Heavy-Tail Distribution Analysis ({name}) ---")

    if G.is_directed():
        degrees = np.array([d for _, d in G.in_degree()])
        x_label = "In-Degree (k)"
    else:
        degrees = np.array([d for _, d in G.degree()])
        x_label = "Degree (k)"

    degrees = degrees[degrees > 0]

    # Guard: Power-law fitting requires a minimum sample size and data variance
    if len(degrees) < 10 or len(np.unique(degrees)) < 2:
        logger.warning(f"Insufficient data points or variance for Power Law fit in {name}.")
        return {}

    try:
        # Optimization: discrete=True for integer-based degree sequences
        fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
        
        alpha = float(fit.power_law.alpha)
        xmin = float(fit.power_law.xmin)

        # Statistical comparison between Power-law and Log-normal models
        try:
            R, p = fit.distribution_compare("power_law", "lognormal")
        except Exception:
            R, p = 0.0, 1.0

        logger.info(f"  Power Law Alpha: {alpha:.4f}")
        logger.info(f"  Xmin (Cutoff):   {xmin}")

        logger.info("[Physics Interpretation]")
        if alpha < 3.5:
            logger.info("  -> Verdict: Distribution is HEAVY-TAILED (Hub-dominated).")
        else:
            logger.info("  -> Verdict: Distribution decays rapidly (No clear hubs).")

        # Precompute PDF curves for visualization (no refit in plotting layer)
        # powerlaw.Fit(..., discrete=True) returns .pdf(x) as an array (length=data);
        # we need one scalar per x for plotting, so take .flat[0].
        x_plot = np.unique(degrees)
        _, counts = np.unique(degrees, return_counts=True)
        pdf_empirical = np.asarray(counts, dtype=float) / len(degrees)
        try:
            pdf_power_law = np.array([
                float(np.atleast_1d(fit.power_law.pdf(x)).flat[0]) for x in x_plot
            ])
            pdf_lognormal = np.array([
                float(np.atleast_1d(fit.lognormal.pdf(x)).flat[0]) for x in x_plot
            ])
        except Exception:
            pdf_power_law = np.full_like(x_plot, np.nan)
            pdf_lognormal = np.full_like(x_plot, np.nan)

        return {
            "alpha": alpha,
            "xmin": xmin,
            "compare_R": float(R),
            "compare_p": float(p),
            "degrees": degrees,
            "x_label": x_label,
            "pdf_x": x_plot,
            "pdf_empirical": pdf_empirical,
            "pdf_power_law": pdf_power_law,
            "pdf_lognormal": pdf_lognormal,
        }
    except (ValueError, RuntimeError, ZeroDivisionError) as e:
        logger.warning(f"Power law fitting mathematically failed for {name}: {e}")
        return {}


def analyze_spectral_properties(G: nx.Graph, name: str = "Network") -> Dict[str, Any]:
    """
    Calculates the Laplacian matrix to understand information diffusion and entropy.
    Isolates the Giant Connected Component (GCC) for statistically valid metrics.
    """
    logger.info("--- Spectral Analysis (Laplacian & Entropy) ---")

    if G.is_directed():
        G_for_spectral = G.to_undirected()
        logger.info("  (Converting directed to undirected for spectral analysis)")
    else:
        G_for_spectral = G.copy()

    _ensure_distance_attribute(G_for_spectral)

    if not nx.is_connected(G_for_spectral):
        logger.warning("Graph disconnected. Using GCC for spectral metrics.")
        G_cc = G_for_spectral.subgraph(max(nx.connected_components(G_for_spectral), key=len)).copy()
    else:
        G_cc = G_for_spectral.copy()

    n = G_cc.number_of_nodes()
    _ensure_distance_attribute(G_cc)

    try:
        avg_path_len = nx.average_shortest_path_length(G_cc, weight="distance")
    except nx.NetworkXError:
        avg_path_len = float(np.log(max(n, 2)))

    logger.info("  Computing Normalized Laplacian and Eigenvalues...")
    L = nx.normalized_laplacian_matrix(G_cc)

    try:
        # Full diagonalization for smaller graphs, sparse approximation for large ones
        if n < 2000:
            eigenvalues = scipy.linalg.eigh(L.todense(), eigvals_only=True)
        else:
            eigenvalues = scipy.sparse.linalg.eigsh(
                L, k=min(n - 1, 100), which="SM", return_eigenvectors=False
            )
    except MemoryError:
        logger.error("Graph too large for diagonalization.")
        return {}

    eigenvalues = np.sort(np.array(eigenvalues, dtype=float))
    lambda_2 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    diffusion_time = float(1.0 / lambda_2) if lambda_2 > 1e-9 else float("inf")

    denom = float(np.sum(eigenvalues))
    if denom <= 0: return {}

    rho = eigenvalues / denom
    vn_entropy = float(-np.sum(rho * np.log(rho + 1e-12)))
    max_entropy = float(np.log(max(n, 2)))

    return {
        "avg_path_length": float(avg_path_len),
        "lambda_2": float(lambda_2),
        "diffusion_time": float(diffusion_time),
        "vn_entropy": float(vn_entropy),
        "max_entropy": float(max_entropy),
        "n_gcc": float(n),
        "eigenvalues": eigenvalues,
    }


def analyze_robustness(G: nx.Graph, name: str = "Network") -> Dict[str, Any]:
    """
    Compares network resilience under random failure vs. targeted hub attacks.
    """
    logger.info("--- Robustness & Percolation Analysis ---")

    G_undirected = G.to_undirected() if G.is_directed() else G

    fraction_to_remove = 0.2
    n_total = G_undirected.number_of_nodes()
    n_remove = int(n_total * fraction_to_remove)
    step_size = max(1, n_remove // 50)

    G_attack = G_undirected.copy()
    target_nodes = [n for n, _ in sorted(G_undirected.degree, key=lambda x: x[1], reverse=True)]
    attack_sizes = [1.0]

    G_random = G_undirected.copy()
    random_targets = list(G_undirected.nodes())
    random.shuffle(random_targets)
    random_sizes = [1.0]

    for i in range(0, n_remove, step_size):
        # Target Attack simulation
        G_attack.remove_nodes_from(target_nodes[i : i + step_size])
        gcc_a = len(max(nx.connected_components(G_attack), key=len)) if len(G_attack) > 0 else 0
        attack_sizes.append(gcc_a / n_total)

        # Random Failure simulation
        G_random.remove_nodes_from(random_targets[i : i + step_size])
        gcc_r = len(max(nx.connected_components(G_random), key=len)) if len(G_random) > 0 else 0
        random_sizes.append(gcc_r / n_total)

    x_axis = np.linspace(0, fraction_to_remove, len(attack_sizes))

    return {
        "random_sizes": random_sizes,
        "attack_sizes": attack_sizes,
        "x_axis": list(x_axis),
        "fraction_to_remove": fraction_to_remove,
    }


def analyze_configuration_model(G: nx.Graph, n_randomizations: int = 10) -> Dict[str, Any]:
    """
    Statistical validation of clustering using randomized null models.
    """
    logger.info("--- Null Model Comparison ---")
    G_cc = G.to_undirected()
    if not nx.is_connected(G_cc):
        G_cc = G_cc.subgraph(max(nx.connected_components(G_cc), key=len)).copy()

    C_real = float(nx.average_clustering(G_cc))
    null_clustering_values: List[float] = []
    n_swaps = 5 * G_cc.number_of_edges()

    for _ in range(n_randomizations):
        G_null = G_cc.copy()
        try:
            nx.double_edge_swap(G_null, nswap=n_swaps, max_tries=n_swaps * 5)
            null_clustering_values.append(float(nx.average_clustering(G_null)))
        except nx.NetworkXError: pass

    if not null_clustering_values: return {}

    avg_null_C = float(np.mean(null_clustering_values))
    std_null_C = float(np.std(null_clustering_values))
    z_score = float((C_real - avg_null_C) / std_null_C) if std_null_C > 1e-9 else 0.0

    return {
        "C_real": C_real,
        "C_null_avg": avg_null_C,
        "z_score": z_score,
        "null_clustering_values": null_clustering_values,
    }
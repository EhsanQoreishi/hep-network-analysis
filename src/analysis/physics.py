import logging
import os
import random
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import powerlaw
import scipy.linalg
import scipy.sparse.linalg
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)


def analyze_power_law(
    G: nx.Graph, name: str = "Network", output_dir: str = "results"
) -> Dict[str, Any]:
    """
    Fits the degree distribution to a Power Law (Scale-Free Network check).

    Physics:
    - Alpha < 3.5 implies a Heavy-Tailed distribution (Hubs exist).
    - Compares Power Law vs Log-Normal fit.

    Args:
        G (nx.Graph): The network graph.
        name (str): Label for the plot.
        output_dir (str): Directory to save results.

    Returns:
        Dict: Contains 'alpha', 'xmin', 'distribution_compare_p_value'.
    """
    logger.info(f"--- Heavy-Tail Distribution Analysis ({name}) ---")

    # Vectorized degree extraction
    degrees = np.array([d for _, d in G.degree()])
    degrees = degrees[degrees > 0]  # Filter zeros

    # Fit the distribution
    fit = powerlaw.Fit(degrees, discrete=True, verbose=False)

    logger.info(f"  Power Law Alpha: {fit.power_law.alpha:.4f}")
    logger.info(f"  Xmin (Cutoff):   {fit.power_law.xmin}")

    # Compare distributions
    R, p = fit.distribution_compare("power_law", "lognormal")

    logger.info("[Physics Interpretation]")
    if fit.power_law.alpha < 3.5:
        logger.info("  -> Verdict: The distribution is HEAVY-TAILED.")
        logger.info(
            "  -> Physical Meaning: The network is dominated by 'Hubs' (Super-Connectors)."
        )
    else:
        logger.info("  -> Verdict: The distribution decays quickly.")

    # Plot
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    fit.plot_pdf(color="b", linear_bins=True, label="Empirical Data")
    fit.power_law.plot_pdf(color="r", linestyle="--", label="Power Law Fit")
    fit.lognormal.plot_pdf(color="g", linestyle="-.", label="Log-Normal Fit")

    plt.title(f"Degree Distribution ({name})")
    plt.xlabel("Degree (k)")
    plt.ylabel("P(k)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(
        output_dir, f"{name.lower().replace('-', '_')}_power_law_fit.png"
    )
    plt.savefig(save_path)
    plt.close()

    return {
        "alpha": fit.power_law.alpha,
        "xmin": fit.power_law.xmin,
        "compare_R": R,
        "compare_p": p,
    }


def analyze_spectral_properties(
    G: nx.Graph, output_dir: str = "results"
) -> Dict[str, float]:
    """
    Analyzes the Eigenvalues of the Laplacian Matrix.

    Metrics:
    - Spectral Gap (lambda_2): Relates to diffusion time (tau ~ 1/lambda_2).
    - Von Neumann Entropy: Measure of structural order vs randomness.

    Returns:
        Dict: Contains 'lambda_2', 'diffusion_time', 'vn_entropy'.
    """
    logger.info("--- Spectral Analysis (Laplacian & Entropy) ---")

    # Spectral analysis requires a connected component
    if not nx.is_connected(G):
        logger.warning(
            "Graph disconnected. Using Giant Connected Component for spectral metrics."
        )
        G_cc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    else:
        G_cc = G.copy()

    n = G_cc.number_of_nodes()

    # 1. Diffusion Time (approx) via Path Length
    try:
        avg_path_len = nx.average_shortest_path_length(G_cc, weight="distance")
    except nx.NetworkXError:
        avg_path_len = np.log(n)
    logger.info(f"  Average Path Length (L): {avg_path_len:.2f}")

    # 2. Diagonalization
    logger.info("  Computing Normalized Laplacian Matrix...")
    L = nx.normalized_laplacian_matrix(G_cc)

    logger.info("  Calculating Eigenvalues...")
    try:
        if n < 2000:
            # Dense solver for small N (exact entropy)
            eigenvalues = scipy.linalg.eigh(L.todense(), eigvals_only=True)
        else:
            # Sparse solver for large N (approximate spectrum)
            logger.info("  (Graph large: approximating spectral density with k=100)")
            eigenvalues = scipy.sparse.linalg.eigsh(
                L, k=min(n - 1, 100), which="SM", return_eigenvectors=False
            )
    except MemoryError:
        logger.error("Graph too large for diagonalization.")
        return {}

    eigenvalues.sort()

    # Algebraic Connectivity
    lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
    diffusion_time = 1.0 / lambda_2 if lambda_2 > 1e-9 else float("inf")

    # Von Neumann Entropy: S = -sum(rho * ln(rho))
    # Normalize eigenvalues to sum to 1
    rho = eigenvalues / np.sum(eigenvalues)
    # Add epsilon to avoid log(0)
    vn_entropy = -np.sum(rho * np.log(rho + 1e-12))
    max_entropy = np.log(n)

    logger.info("Spectral Metrics:")
    logger.info(f"  Algebraic Connectivity (lambda_2): {lambda_2:.6f}")
    logger.info(f"  Diffusion Time (tau): approx {diffusion_time:.2f} steps")
    logger.info(f"  Von Neumann Entropy (S): {vn_entropy:.4f} (Max: {max_entropy:.4f})")

    # Plot Spectral Density
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    try:
        kde = gaussian_kde(eigenvalues)
        x_range = np.linspace(min(eigenvalues), max(eigenvalues), 200)
        plt.plot(x_range, kde(x_range), color="blue", lw=2, label="Spectral Density")
        plt.fill_between(x_range, kde(x_range), color="blue", alpha=0.1)
    except Exception:
        plt.hist(eigenvalues, bins=50, color="blue", alpha=0.5)

    plt.title(f"Spectral Density (Entropy S={vn_entropy:.2f})")
    plt.xlabel(r"Eigenvalue ($\lambda$)")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(output_dir, "spectral_density_entropy.png"))
    plt.close()

    return {
        "lambda_2": lambda_2,
        "diffusion_time": diffusion_time,
        "vn_entropy": vn_entropy,
    }


def analyze_robustness(
    G: nx.Graph, output_dir: str = "results"
) -> Dict[str, List[float]]:
    """
    Percolation Analysis: Simulates node removal to test network fragility.
    Compares Random Failure vs Targeted Attack (removing hubs).

    Returns:
        Dict: Lists of GCC sizes for 'random' and 'attack' scenarios.
    """
    logger.info("--- Robustness & Perturbation Analysis ---")

    fraction_to_remove = 0.2
    steps = 20
    n_total = G.number_of_nodes()
    n_remove = int(n_total * fraction_to_remove)
    step_size = max(1, n_remove // steps)

    logger.info(f"  Simulating removal of {int(fraction_to_remove * 100)}% nodes...")

    # Strategy 1: Targeted Attack (Hubs first)
    G_attack = G.copy()
    targets = sorted(G.degree, key=lambda x: x[1], reverse=True)
    target_nodes = [n for n, _ in targets]
    attack_sizes = [1.0]

    # Strategy 2: Random Failure
    G_random = G.copy()
    random_targets = list(G.nodes())
    random.shuffle(random_targets)
    random_sizes = [1.0]

    # Simulation Loop
    # Note: range arguments must be int
    for i in range(0, n_remove, step_size):
        # Attack
        batch_attack = target_nodes[i : i + step_size]
        G_attack.remove_nodes_from(batch_attack)

        if len(G_attack) > 0:
            gcc_a = len(max(nx.connected_components(G_attack), key=len))
            attack_sizes.append(gcc_a / n_total)
        else:
            attack_sizes.append(0.0)

        # Random
        batch_random = random_targets[i : i + step_size]
        G_random.remove_nodes_from(batch_random)

        if len(G_random) > 0:
            gcc_r = len(max(nx.connected_components(G_random), key=len))
            random_sizes.append(gcc_r / n_total)
        else:
            random_sizes.append(0.0)

    # Plot
    x_axis = np.linspace(0, fraction_to_remove, len(attack_sizes))

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, random_sizes, "g-o", label="Random Failure")
    plt.plot(x_axis, attack_sizes, "r-s", label="Targeted Attack (Hubs)")

    plt.title("Percolation Threshold: Network Robustness")
    plt.xlabel("Fraction of Nodes Removed ($f$)")
    plt.ylabel("Giant Component Size ($S$)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(output_dir, "network_robustness.png"))
    plt.close()

    return {"random_sizes": random_sizes, "attack_sizes": attack_sizes}


def analyze_configuration_model(
    G: nx.Graph, n_randomizations: int = 10, output_dir: str = "results"
) -> Dict[str, float]:
    """
    Compares real clustering to a randomized Configuration Model (Null Model).

    Returns:
        Dict: Contains 'C_real', 'C_null_avg', 'z_score'.
    """
    logger.info("--- Null Model Comparison ---")

    if not nx.is_connected(G):
        G_cc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    else:
        G_cc = G.copy()

    C_real = nx.average_clustering(G_cc)

    null_clustering_values = []
    # Heuristic for sufficient mixing
    n_swaps = 5 * G_cc.number_of_edges()

    # Generate Null Models via Edge Swapping
    for _ in range(n_randomizations):
        G_null = G_cc.copy()
        try:
            nx.double_edge_swap(G_null, nswap=n_swaps, max_tries=n_swaps * 5)
            null_clustering_values.append(nx.average_clustering(G_null))
        except nx.NetworkXError:
            pass

    if not null_clustering_values:
        logger.warning("Null model generation failed (graph might be too small/dense).")
        return {}

    avg_null_C = np.mean(null_clustering_values)
    std_null_C = np.std(null_clustering_values)
    z_score = (C_real - avg_null_C) / std_null_C if std_null_C > 1e-9 else 0.0

    logger.info(f"  Real Clustering: {C_real:.4f}")
    logger.info(f"  Null Model <C>:  {avg_null_C:.4f}")
    logger.info(f"  Z-Score:         {z_score:.2f}")

    # Plot
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(null_clustering_values, color="gray", alpha=0.7, label="Null Model")
    plt.axvline(
        C_real, color="red", linestyle="dashed", linewidth=2, label="Real Network"
    )
    plt.title(f"Null Model Comparison (Z={z_score:.2f})")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "configuration_model_comparison.png"))
    plt.close()

    return {"C_real": C_real, "C_null_avg": avg_null_C, "z_score": z_score}

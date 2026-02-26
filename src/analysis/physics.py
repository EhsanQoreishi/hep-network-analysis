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

def _sanitize_name(name: str) -> str:
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
    base, ext = os.path.splitext(path)
    if ext.lower() != ".pdf":
        path = base + ".pdf"
    plt.savefig(path, format="pdf", bbox_inches="tight")
    logger.info(f"Saved figure to: {path}")

def _ensure_distance_attribute(G: nx.Graph) -> None:
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

def analyze_power_law(G: nx.Graph, name: str = "Network", output_dir: str = "results") -> Dict[str, Any]:
    logger.info(f"--- Heavy-Tail Distribution Analysis ({name}) ---")

    if G.is_directed():
        degrees = np.array([d for _, d in G.in_degree()])
        x_label = "In-Degree (k)"
    else:
        degrees = np.array([d for _, d in G.degree()])
        x_label = "Degree (k)"

    degrees = degrees[degrees > 0]

    if len(degrees) < 10:
        logger.warning(f"Not enough data points for Power Law fit in {name}.")
        return {}

    fit = powerlaw.Fit(degrees, discrete=True, verbose=False)

    logger.info(f"  Power Law Alpha: {fit.power_law.alpha:.4f}")
    logger.info(f"  Xmin (Cutoff):   {fit.power_law.xmin}")

    R, p = fit.distribution_compare("power_law", "lognormal")

    logger.info("[Physics Interpretation]")
    if fit.power_law.alpha < 3.5:
        logger.info("  -> Verdict: The distribution is HEAVY-TAILED.")
        logger.info("  -> Physical Meaning: The network is dominated by 'Hubs' (Super-Connectors).")
    else:
        logger.info("  -> Verdict: The distribution decays quickly.")

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    fit.plot_pdf(color="b", linear_bins=True, label="Empirical Data")
    fit.power_law.plot_pdf(color="r", linestyle="--", label="Power Law Fit")
    fit.lognormal.plot_pdf(color="g", linestyle="-.", label="Log-Normal Fit")

    plt.title(f"Degree Distribution ({name})")
    plt.xlabel(x_label)
    plt.ylabel("P(k)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    safe_name = _sanitize_name(name)
    save_path = os.path.join(output_dir, f"{safe_name}_degree_distribution_powerlaw.pdf")
    _savefig(save_path)
    plt.close()

    return {
        "alpha": float(fit.power_law.alpha),
        "xmin": float(fit.power_law.xmin),
        "compare_R": float(R),
        "compare_p": float(p),
    }

def analyze_spectral_properties(G: nx.Graph, name: str = "Network", output_dir: str = "results") -> Dict[str, float]:
    logger.info("--- Spectral Analysis (Laplacian & Entropy) ---")

    if G.is_directed():
        G_for_spectral = G.to_undirected()
        logger.info("  (Converting directed graph to undirected for spectral analysis)")
    else:
        G_for_spectral = G.copy()

    _ensure_distance_attribute(G_for_spectral)

    if not nx.is_connected(G_for_spectral):
        logger.warning("Graph disconnected. Using Giant Connected Component for spectral metrics.")
        G_cc = G_for_spectral.subgraph(max(nx.connected_components(G_for_spectral), key=len)).copy()
    else:
        G_cc = G_for_spectral.copy()

    n = G_cc.number_of_nodes()

    _ensure_distance_attribute(G_cc)

    try:
        avg_path_len = nx.average_shortest_path_length(G_cc, weight="distance")
    except nx.NetworkXError:
        avg_path_len = float(np.log(max(n, 2)))
    logger.info(f"  Average Path Length (L, weighted): {avg_path_len:.4f}")

    logger.info("  Computing Normalized Laplacian Matrix...")
    L = nx.normalized_laplacian_matrix(G_cc)

    logger.info("  Calculating Eigenvalues...")
    try:
        if n < 2000:
            eigenvalues = scipy.linalg.eigh(L.todense(), eigvals_only=True)
        else:
            logger.info("  (Graph large: approximating spectral density with k=100)")
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
    if denom <= 0:
        return {}

    rho = eigenvalues / denom
    vn_entropy = float(-np.sum(rho * np.log(rho + 1e-12)))
    max_entropy = float(np.log(max(n, 2)))

    logger.info("Spectral Metrics:")
    logger.info(f"  Algebraic Connectivity (lambda_2): {lambda_2:.6f}")
    logger.info(f"  Diffusion Time (tau): approx {diffusion_time:.2f} steps")
    logger.info(f"  Von Neumann Entropy (S): {vn_entropy:.4f} (Max: {max_entropy:.4f})")

    os.makedirs(output_dir, exist_ok=True)
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

    safe_name = _sanitize_name(name)
    save_path = os.path.join(output_dir, f"{safe_name}_spectral_density_entropy.pdf")
    _savefig(save_path)
    plt.close()

    return {
        "avg_path_length": float(avg_path_len),
        "lambda_2": float(lambda_2),
        "diffusion_time": float(diffusion_time),
        "vn_entropy": float(vn_entropy),
        "max_entropy": float(max_entropy),
        "n_gcc": float(n),
    }

def analyze_robustness(G: nx.Graph, name: str = "Network", output_dir: str = "results") -> Dict[str, List[float]]:
    logger.info("--- Robustness & Perturbation Analysis ---")

    if G.is_directed():
        logger.info("  (Converting directed graph to undirected for robustness analysis)")
        G_undirected = G.to_undirected()
    else:
        G_undirected = G

    fraction_to_remove = 0.2
    n_total = G_undirected.number_of_nodes()
    n_remove = int(n_total * fraction_to_remove)

    steps_to_simulate = 50
    step_size = max(1, n_remove // steps_to_simulate)

    logger.info(f"  Simulating removal of {int(fraction_to_remove * 100)}% nodes...")
    logger.info(f"  Optimization: Computing GCC every {step_size} removals.")

    G_attack = G_undirected.copy()
    targets = sorted(G_undirected.degree, key=lambda x: x[1], reverse=True)
    target_nodes = [n for n, _ in targets]
    attack_sizes = [1.0]

    G_random = G_undirected.copy()
    random_targets = list(G_undirected.nodes())
    random.shuffle(random_targets)
    random_sizes = [1.0]

    for i in range(0, n_remove, step_size):
        batch_attack = target_nodes[i : i + step_size]
        G_attack.remove_nodes_from(batch_attack)
        if len(G_attack) > 0:
            gcc_a = len(max(nx.connected_components(G_attack), key=len))
            attack_sizes.append(gcc_a / n_total)
        else:
            attack_sizes.append(0.0)

        batch_random = random_targets[i : i + step_size]
        G_random.remove_nodes_from(batch_random)
        if len(G_random) > 0:
            gcc_r = len(max(nx.connected_components(G_random), key=len))
            random_sizes.append(gcc_r / n_total)
        else:
            random_sizes.append(0.0)

    x_axis = np.linspace(0, fraction_to_remove, len(attack_sizes))

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8.5, 6))
    plt.plot(x_axis, random_sizes, "g-o", label="Random Failure")
    plt.plot(x_axis, attack_sizes, "r-s", label="Targeted Attack (Degree hubs)")
    plt.title(f"Network Robustness ({name})")
    plt.xlabel("Fraction of nodes removed (f)")
    plt.ylabel("Giant component size (S)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    safe_name = _sanitize_name(name)
    save_path = os.path.join(output_dir, f"{safe_name}_network_robustness.pdf")
    _savefig(save_path)
    plt.close()

    return {"random_sizes": random_sizes, "attack_sizes": attack_sizes}

def analyze_configuration_model(
    G: nx.Graph, name: str = "Network", n_randomizations: int = 10, output_dir: str = "results"
) -> Dict[str, float]:
    logger.info("--- Null Model Comparison ---")
    logger.info(f"  Randomizations: {int(n_randomizations)}")

    if G.is_directed():
        logger.info("  (Using to_undirected() for configuration model comparison)")
        G_undirected = G.to_undirected()
    else:
        G_undirected = G

    if not nx.is_connected(G_undirected):
        G_cc = G_undirected.subgraph(max(nx.connected_components(G_undirected), key=len)).copy()
    else:
        G_cc = G_undirected.copy()

    C_real = float(nx.average_clustering(G_cc))

    null_clustering_values: List[float] = []
    n_swaps = 5 * G_cc.number_of_edges()
    logger.info(f"  Edge swaps per randomization: {int(n_swaps)}")

    for _ in range(n_randomizations):
        G_null = G_cc.copy()
        try:
            nx.double_edge_swap(G_null, nswap=n_swaps, max_tries=n_swaps * 5)
            null_clustering_values.append(float(nx.average_clustering(G_null)))
        except nx.NetworkXError:
            pass

    if not null_clustering_values:
        logger.warning("Null model generation failed (graph might be too small/dense).")
        return {}

    avg_null_C = float(np.mean(null_clustering_values))
    std_null_C = float(np.std(null_clustering_values))
    z_score = float((C_real - avg_null_C) / std_null_C) if std_null_C > 1e-9 else 0.0

    logger.info(f"  Real Clustering: {C_real:.4f}")
    logger.info(f"  Null Model <C>:  {avg_null_C:.4f}")
    logger.info(f"  Z-Score:         {z_score:.2f}")

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(null_clustering_values, color="gray", alpha=0.7, label="Null Model")
    plt.axvline(C_real, color="red", linestyle="dashed", linewidth=2, label="Real Network")
    plt.title(f"Configuration-Model Clustering ({name}) (Z={z_score:.2f})")
    plt.xlabel("Average clustering coefficient")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.25)

    safe_name = _sanitize_name(name)
    save_path = os.path.join(output_dir, f"{safe_name}_configuration_model_clustering.pdf")
    _savefig(save_path)
    plt.close()

    return {"C_real": C_real, "C_null_avg": avg_null_C, "z_score": z_score}

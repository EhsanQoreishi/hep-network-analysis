import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from typing import Dict, List, Optional, Tuple

def print_global_metrics(G: nx.Graph) -> None:
    """
    Computes and prints standard topological metrics.
    
    Metrics:
    - Density: Ratio of actual edges to possible edges.
    - Transitivity: Global clustering (fraction of closed triplets).
    - Avg Clustering: Mean of local clustering coefficients.
    """
    print("\n--- Global Graph Metrics ---")
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    print(f"Nodes: {n_nodes}, Edges: {n_edges}")
    
    density = nx.density(G)
    print(f"Edge Density: {density:.6f}")

    transitivity = nx.transitivity(G)
    print(f"Global Clustering Coeff (Transitivity): {transitivity:.4f}")

    avg_clustering = nx.average_clustering(G)
    print(f"Average Clustering Coefficient: {avg_clustering:.4f}")

def print_top_authors(G_co: nx.Graph, G_cit: nx.DiGraph) -> None:
    """
    Identifies top authors by Degree (Social) and In-Degree (Citation).
    Also calculates Betweenness Centrality to find 'bridges'.
    """
    print("\n--- Centrality Analysis ---")

    # 1. Degree Centrality (Hubs)
    top_connected = sorted(
        G_co.degree(), key=lambda x: x[1], reverse=True
    )[:5]
    print("Most Collaborative (High Degree):")
    for author, degree in top_connected:
        print(f"  - {author}: {degree} co-authors")

    # 2. Citation Influence (Authorities)
    # Using 'weight' accounts for the strength of citation flow
    top_cited = sorted(
        G_cit.in_degree(weight="weight"), key=lambda x: x[1], reverse=True
    )[:5]
    print("\nMost Influential (High Citations):")
    for author, count in top_cited:
        print(f"  - {author}: {count:.1f} citations")

    # 3. Betweenness Centrality (Bridges)
    print("\nCalculating Betweenness Centrality (this may take a moment)...")
    try:
        # We use 'distance' (1/shared_papers) because betweenness looks for shortest paths.
        # k=500 approximation is used for speed on large graphs. Remove k for exact results.
        betweenness = nx.betweenness_centrality(G_co, weight='distance', k=500)
        
        top_bridges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top Bridges (High Betweenness - Interdisciplinary Connectors):")
        for author, score in top_bridges:
            print(f"  - {author}: {score:.4f}")
            
    except KeyError:
        print("  [Error] 'distance' attribute missing on edges.")

def analyze_layer_shortest_paths(G_cit: nx.DiGraph, G_co: nx.Graph, output_dir: str = "results") -> float:
    """
    Calculates the social distance (co-authorship path) between authors 
    who have a direct citation link.
    """
    print("\n--- Cross-Layer Path Analysis ---")
    distances = []
    
    # Analyze citation edges to see if they follow social lines
    valid_pairs = 0
    for u, v in G_cit.edges():
        if u in G_co and v in G_co:
            try:
                # Weighted Shortest Path in Social Layer
                d = nx.shortest_path_length(G_co, source=u, target=v, weight='distance')
                distances.append(d)
                valid_pairs += 1
            except nx.NetworkXNoPath:
                distances.append(-1)

    reachable_distances = [d for d in distances if d != -1]
    avg_dist = np.mean(reachable_distances) if reachable_distances else 0.0

    print(f"Analyzed {valid_pairs} pairs connected in Citation layer.")
    print(f"Average Co-authorship distance for these pairs: {avg_dist:.2f}")

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(
        reachable_distances, 
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

def analyze_strength_distribution(G: nx.Graph, name: str = "Network", output_dir: str = "results") -> None:
    """
    Analyzes the correlation between Node Strength (s) and Degree (k).
    s ~ k^beta. 
    beta > 1 implies 'Rich-Club' behavior (hubs have stronger ties).
    """
    print(f"\n--- Weighted Strength Analysis ({name}) ---")
    
    strengths = dict(G.degree(weight='weight'))
    degrees = dict(G.degree(weight=None))
    
    s_values = np.array(list(strengths.values()))
    k_values = np.array([degrees[n] for n in strengths.keys()])
    
    # Filter zeros
    valid_mask = (k_values > 0) & (s_values > 0)
    k_values = k_values[valid_mask]
    s_values = s_values[valid_mask]

    # Calculate average strength per degree class for clean fitting
    k_unique = np.unique(k_values)
    s_avg_k = []
    for k in k_unique:
        mean_s = np.mean(s_values[k_values == k])
        s_avg_k.append(mean_s)
    
    # Fit Power Law: s ~ A * k^beta
    log_k = np.log10(k_unique)
    log_s = np.log10(s_avg_k)
    beta, intercept = np.polyfit(log_k, log_s, 1)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(k_values, s_values, alpha=0.1, color='gray', s=10, label='Nodes')
    plt.loglog(k_unique, s_avg_k, 'bo', label='Average <s(k)>')
    
    fit_y = 10**intercept * k_unique**beta
    plt.plot(k_unique, fit_y, 'r--', linewidth=2, label=f'Fit: $\\beta = {beta:.2f}$')
    
    plt.title(f"Strength vs Degree Correlation")
    plt.xlabel("Degree k")
    plt.ylabel("Strength s")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, f"{name.lower()}_strength_degree_correlation.png")
    plt.savefig(save_path)
    plt.close()
    
    print(f"  Fit exponent (beta): {beta:.4f}")
    if beta > 1.1:
        print(f"  -> Super-linear (beta > 1): Hubs work harder/more intensely.")

def analyze_multiplex_correlation(G_co: nx.Graph, G_cit: nx.DiGraph, output_dir: str = "results") -> None:
    """
    Correlates Citation Influence (PageRank) with Social Brokerage (Betweenness).
    """
    print("\n--- Multiplex Correlation Analysis ---")

    common_authors = list(set(G_co.nodes()).intersection(set(G_cit.nodes())))
    if len(common_authors) < 10:
        print("Not enough common authors for correlation.")
        return

    # Calculate metrics
    pagerank = nx.pagerank(G_cit, weight="weight")
    # k=None for exact, or int for approx
    betweenness = nx.betweenness_centrality(G_co, weight="distance", k=min(len(common_authors), 500))

    x_data = [pagerank.get(a, 0) for a in common_authors]
    y_data = [betweenness.get(a, 0) for a in common_authors]

    corr, p_value = spearmanr(x_data, y_data)

    print(f"  Spearman Correlation: {corr:.4f} (p={p_value:.4e})")

    # Hexbin Plot for density
    x_plot = [x for x, y in zip(x_data, y_data) if x > 0 and y > 0]
    y_plot = [y for x, y in zip(x_data, y_data) if x > 0 and y > 0]

    plt.figure(figsize=(10, 7))
    hb = plt.hexbin(
        x_plot, y_plot,
        gridsize=30, cmap="inferno", bins="log", xscale="log", yscale="log"
    )
    plt.colorbar(hb, label="log10(Count)")
    plt.xlabel("Citation Influence (PageRank)")
    plt.ylabel("Social Brokerage (Betweenness)")
    plt.title(f"Multiplex Correlation (r={corr:.2f})")
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(output_dir, "multiplex_centrality_correlation.png"))
    plt.close()
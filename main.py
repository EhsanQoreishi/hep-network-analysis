"""
Command-line orchestrator for the HEP-Th network analysis pipeline.

This module wires together the preprocessing, network construction, structural
and physics analysis, and visualization layers. It parses CLI arguments, calls
into ``src.analysis.*`` for computations, and then hands the returned data to
``src.visualization`` to produce plots and HTML outputs.
"""

import argparse
import csv
import logging
import os
import sys
from typing import Dict, List, Set

import networkx as nx

from src.analysis.communities import (
    analyze_communities_robust,
    check_community_distribution,
    get_community_partition,
)
from src.analysis.physics import (
    analyze_configuration_model,
    analyze_power_law,
    analyze_robustness,
    analyze_spectral_properties,
)
from src.analysis.structural import (
    analyze_degree_correlation,
    analyze_layer_shortest_paths,
    analyze_multiplex_correlation,
    analyze_strength_distribution,
    export_centrality_tables,
    get_global_metrics,
    get_top_authors,
)
from src.networks import build_networks
from src.preprocessing import parse_abstracts

from src.visualization import (
    plot_community_distribution,
    plot_configuration_model,
    plot_cross_layer_paths,
    plot_degree_vs_instrength,
    plot_multiplex_correlation,
    plot_power_law,
    plot_robustness,
    plot_spectral_density,
    plot_strength_distribution,
    plot_top_centralities,
    visualize_network,
)


def setup_logging(debug_mode: bool = False):
    """
    Configures dual-level logging to keep the console clean while saving debug data to files.
    """
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    simple_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    verbose_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    info_handler = logging.FileHandler("logs/run_info.log", mode="w")
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(simple_formatter)
    logger.addHandler(info_handler)

    debug_handler = logging.FileHandler("logs/run_debug.log", mode="w")
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(verbose_formatter)
    logger.addHandler(debug_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_level = logging.DEBUG if debug_mode else logging.INFO
    console_handler.setLevel(console_level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    logging.info("Logging initialized: Info -> logs/run_info.log | Debug -> logs/run_debug.log")


def write_topics_to_csv(topics: List[Dict], layer_name: str, output_dir: str):
    """
    Helper function to save the NLP-extracted community topics to a CSV.
    This keeps file I/O separate from the pure math in communities.py.
    """
    if not topics:
        return
        
    safe_layer = layer_name.strip().lower().replace(" ", "_").replace("-", "_")
    csv_path = os.path.join(output_dir, f"{safe_layer}_top_communities_tfidf.csv")
    
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "community_id", "community_size", "keywords"])
        for t in topics:
            writer.writerow([t["layer"], t["community_id"], t["community_size"], ", ".join(t["keywords"])])
            
    logging.getLogger("HEP-Analysis").info(f"Saved topics table to: {csv_path}")


def main():
    """
    Entry point for the CLI tool.

    Parses command-line arguments, runs all pipeline phases (ETL, network build,
    structural + physics analysis, community detection, and visualization), and
    writes results into the configured output directory.
    """
    parser = argparse.ArgumentParser(description="Analyze citation and co-authorship networks from HEP-Th data.")
    parser.add_argument("--data", type=str, default="data/cit-HepTh.txt", help="Path to the citation edges text file.")
    parser.add_argument("--abstracts", type=str, default="data/cit-HepTh-abstracts", help="Directory containing abstract files (.abs).")
    parser.add_argument("--output", type=str, default="results", help="Directory to save output plots and HTML files.")
    parser.add_argument("--jobs", type=int, default=-1, help="Number of CPU cores for parallel processing (-1 = all).")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")

    args = parser.parse_args()
    setup_logging(args.debug)
    logger = logging.getLogger("HEP-Analysis")

    os.makedirs(args.output, exist_ok=True)
    logger.info(f"Results will be saved to: {args.output}")

    # =========================================================================
    # Phase 1 & 2: Data Ingestion and Graph Building
    # =========================================================================
    logger.info(f"[Phase 1] Parsing Abstracts from {args.abstracts} (Jobs: {args.jobs})...")
    if not os.path.exists(args.abstracts) or not os.path.exists(args.data):
        logger.error("Data files not found. Please check your paths.")
        sys.exit(1)

    paper_to_authors, paper_to_text, author_to_papers = parse_abstracts(args.abstracts, n_jobs=args.jobs)

    logger.info("[Phase 2] Building Networks...")
    G_co, G_cit = build_networks(args.data, paper_to_authors)

    if G_co.number_of_nodes() == 0:
        logger.error("Failed to build Co-authorship network. Exiting.")
        sys.exit(1)

    # =========================================================================
    # Phase 3: Alignment
    # =========================================================================
    logger.info("[Phase 3] Aligning Network Layers...")
    common_nodes = set(G_co.nodes()) & set(G_cit.nodes())

    if not common_nodes:
        logger.error("No overlap found between Citation and Co-authorship layers.")
        sys.exit(1)

    G_co_core = G_co.subgraph(common_nodes).copy()
    largest_cc_nodes: Set[str] = max(nx.connected_components(G_co_core), key=len)

    G_social = G_co.subgraph(largest_cc_nodes).copy()
    G_intellectual = G_cit.subgraph(largest_cc_nodes).copy()

    logger.info(f"  Original Nodes: {G_co.number_of_nodes()}")
    logger.info(f"  Intersection Nodes: {len(common_nodes)}")
    logger.info(f"  Final Analysis Set (GCC): {G_social.number_of_nodes()} authors")

    # =========================================================================
    # Phase 4: Structural & Physics Analysis
    # We now fetch raw data first, THEN pass it to the plotter.
    # =========================================================================
    logger.info("[Phase 4] Running Structural & Physics Analysis...")

    logger.info("--- Global Metrics: Social Layer ---")
    get_global_metrics(G_social)
    logger.info("--- Global Metrics: Citation Layer ---")
    get_global_metrics(G_intellectual)

    get_top_authors(G_social, G_intellectual)

    # Centralities
    cent_data = export_centrality_tables(G_social, output_dir=args.output, top_n=10)
    plot_top_centralities(cent_data, output_dir=args.output, top_n=10)

    # Degree vs In-Strength
    deg_corr_data = analyze_degree_correlation(G_social, G_intellectual)
    plot_degree_vs_instrength(deg_corr_data, output_dir=args.output)

    # Strength Distribution
    str_soc = analyze_strength_distribution(G_social, name="Social-Layer")
    plot_strength_distribution(str_soc, name="Social-Layer", output_dir=args.output)
    
    str_cit = analyze_strength_distribution(G_intellectual, name="Citation-Layer")
    plot_strength_distribution(str_cit, name="Citation-Layer", output_dir=args.output)

    # Power Law
    pl_soc = analyze_power_law(G_social, name="Social-Layer")
    plot_power_law(pl_soc, name="Social-Layer", output_dir=args.output)
    
    pl_cit = analyze_power_law(G_intellectual, name="Citation-Layer")
    plot_power_law(pl_cit, name="Citation-Layer", output_dir=args.output)

    logger.info("--- Physics Analysis: Social Layer ---")
    spec_soc = analyze_spectral_properties(G_social, name="Social-Layer")
    plot_spectral_density(spec_soc, name="Social-Layer", output_dir=args.output)
    
    rob_soc = analyze_robustness(G_social, name="Social-Layer")
    plot_robustness(rob_soc, name="Social-Layer", output_dir=args.output)
    
    cm_soc = analyze_configuration_model(G_social)
    plot_configuration_model(cm_soc, name="Social-Layer", output_dir=args.output)

    logger.info("--- Physics Analysis: Citation Layer ---")
    spec_cit = analyze_spectral_properties(G_intellectual, name="Citation-Layer")
    plot_spectral_density(spec_cit, name="Citation-Layer", output_dir=args.output)
    
    rob_cit = analyze_robustness(G_intellectual, name="Citation-Layer")
    plot_robustness(rob_cit, name="Citation-Layer", output_dir=args.output)
    
    cm_cit = analyze_configuration_model(G_intellectual)
    plot_configuration_model(cm_cit, name="Citation-Layer", output_dir=args.output)

    # Cross-layer & Multiplex
    cross_layer_data = analyze_layer_shortest_paths(G_intellectual, G_social)
    plot_cross_layer_paths(cross_layer_data, output_dir=args.output)

    multi_corr_data = analyze_multiplex_correlation(G_social, G_intellectual)
    plot_multiplex_correlation(multi_corr_data, output_dir=args.output)

    # =========================================================================
    # Phase 5: Communities & Topics
    # =========================================================================
    logger.info(f"[Phase 5] Detecting Communities & Topics (Jobs: {args.jobs})...")

    # Social Layer Communities
    comm_soc_data = analyze_communities_robust(
        G_social, author_to_papers, paper_to_text, layer_name="Social-Layer", n_iterations=5, n_jobs=args.jobs
    )
    write_topics_to_csv(comm_soc_data.get("topics", []), "Social-Layer", args.output)

    # Citation Layer Communities
    comm_cit_data = analyze_communities_robust(
        G_intellectual, author_to_papers, paper_to_text, layer_name="Citation-Layer", n_iterations=5, n_jobs=args.jobs
    )
    write_topics_to_csv(comm_cit_data.get("topics", []), "Citation-Layer", args.output)

    # Community Size Distributions
    dist_soc = check_community_distribution(G_social, layer_name="Social-Layer")
    plot_community_distribution(dist_soc, layer_name="Social-Layer", output_dir=args.output)
    
    dist_cit = check_community_distribution(G_intellectual, layer_name="Citation-Layer")
    plot_community_distribution(dist_cit, layer_name="Citation-Layer", output_dir=args.output)

    # =========================================================================
    # Phase 6: Visualization
    # =========================================================================
    logger.info("[Phase 6] Generating Interactive Map...")
    degrees_social = dict(G_social.degree())
    top_nodes = sorted(degrees_social, key=degrees_social.get, reverse=True)[:500]
    G_map = G_social.subgraph(top_nodes).copy()
    partition = get_community_partition(G_map, random_state=0)
    visualize_network(
        G_map,
        title=os.path.join(args.output, "interactive_map.html"),
        partition=partition,
        degree_full=degrees_social,
    )

    logger.info(f"Done! All results saved to: {args.output}")


if __name__ == "__main__":
    main()
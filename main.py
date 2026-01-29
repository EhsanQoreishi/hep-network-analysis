import argparse
import logging
import os
import sys
from typing import Set

import networkx as nx

from src.analysis.communities import analyze_communities_robust
from src.analysis.physics import (
    analyze_configuration_model,
    analyze_power_law,
    analyze_robustness,
    analyze_spectral_properties,
)
from src.analysis.structural import (
    analyze_layer_shortest_paths,
    analyze_multiplex_correlation,
    analyze_strength_distribution,
    get_global_metrics,
    get_top_authors,
)
from src.networks import build_networks
from src.preprocessing import parse_abstracts
from src.visualization import visualize_network


def setup_logging(debug_mode: bool = False) -> None:
    """Configures the logging format and level."""
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    """
    Main execution pipeline for HEP-Th Network Analysis.
    Orchestrates ETL, Network Construction, Physics Analysis, and Visualization.
    """
    # 1. CLI Argument Parsing
    parser = argparse.ArgumentParser(
        description="Analyze citation and co-authorship networks from HEP-Th data."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/cit-HepTh.txt",
        help="Path to the citation edges text file.",
    )
    parser.add_argument(
        "--abstracts",
        type=str,
        default="data/cit-HepTh-abstracts",
        help="Directory containing abstract files (.abs).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Directory to save output plots and HTML files.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose debug logging."
    )

    args = parser.parse_args()
    setup_logging(args.debug)
    logger = logging.getLogger("HEP-Analysis")

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    logger.info(f"Results will be saved to: {args.output}")

    # 2. Data Parsing (ETL)
    logger.info(f"[Phase 1] Parsing Abstracts from {args.abstracts}...")
    if not os.path.exists(args.abstracts) or not os.path.exists(args.data):
        logger.error("Data files not found. Please check your paths.")
        sys.exit(1)

    paper_to_authors, paper_to_text, author_to_papers = parse_abstracts(args.abstracts)

    # 3. Network Construction
    logger.info("[Phase 2] Building Networks...")
    G_co, G_cit = build_networks(args.data, paper_to_authors)

    if G_co.number_of_nodes() == 0:
        logger.error("Failed to build Co-authorship network. Exiting.")
        sys.exit(1)

    # 4. Alignment & GCC Extraction
    logger.info("[Phase 3] Aligning Network Layers...")
    common_nodes = set(G_co.nodes()) & set(G_cit.nodes())

    if not common_nodes:
        logger.error("No overlap found between Citation and Co-authorship layers.")
        sys.exit(1)

    # Extract the Giant Connected Component (GCC) from the core set
    # This ensures spectral properties (eigenvalues) are well-defined
    G_co_core = G_co.subgraph(common_nodes).copy()
    largest_cc_nodes: Set[str] = max(nx.connected_components(G_co_core), key=len)

    G_social = G_co.subgraph(largest_cc_nodes).copy()
    G_intellectual = G_cit.subgraph(largest_cc_nodes).copy()

    logger.info(f"  Original Nodes: {G_co.number_of_nodes()}")
    logger.info(f"  Intersection Nodes: {len(common_nodes)}")
    logger.info(f"  Final Analysis Set (GCC): {G_social.number_of_nodes()} authors")

    # 5. Physics & Statistical Analysis
    logger.info("[Phase 4] Running Structural & Physics Analysis...")

    # Structural Metrics
    get_global_metrics(G_social)
    get_top_authors(G_social, G_intellectual)
    analyze_strength_distribution(G_social, name="Social-Layer", output_dir=args.output)

    # Complex Systems Physics
    analyze_power_law(G_social, name="Social-Layer", output_dir=args.output)
    analyze_spectral_properties(G_social, output_dir=args.output)
    analyze_robustness(G_social, output_dir=args.output)
    analyze_configuration_model(G_social, output_dir=args.output)

    # Cross-Layer Dynamics
    analyze_layer_shortest_paths(G_intellectual, G_social, output_dir=args.output)
    analyze_multiplex_correlation(G_social, G_intellectual, output_dir=args.output)

    # 6. Community Detection (NLP + Topology)
    logger.info("[Phase 5] Detecting Communities & Topics...")
    analyze_communities_robust(
        G_social, author_to_papers, paper_to_text, n_iterations=5
    )

    # 7. Visualization
    logger.info("[Phase 6] Generating Interactive Map...")
    visualize_network(G_social, title=os.path.join(args.output, "interactive_map.html"))

    logger.info(f"Done! All results saved to: {args.output}")


if __name__ == "__main__":
    main()

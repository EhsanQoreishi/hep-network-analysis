import os
import argparse
import networkx as nx

# Import modules from your new package structure
from src.preprocessing import parse_abstracts
from src.networks import build_networks
from src.visualization import visualize_network

# Import analysis functions from the sub-package
from src.analysis import (
    print_global_metrics,
    analyze_power_law,
    analyze_spectral_properties,
    analyze_robustness,
    analyze_communities_robust,
    analyze_multiplex_correlation,
    print_top_authors,
    analyze_strength_distribution,
    analyze_configuration_model,
    analyze_layer_shortest_paths
)

def main():
    """
    Main execution pipeline for HEP-Th Network Analysis.
    """
    # 1. CLI Argument Parsing
    parser = argparse.ArgumentParser(
        description="Analyze citation and co-authorship networks from HEP-Th data."
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default="data/cit-HepTh.txt",
        help="Path to the citation edges text file."
    )
    parser.add_argument(
        "--abstracts", 
        type=str, 
        default="data/cit-HepTh-abstracts",
        help="Directory containing abstract files (.abs)."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="results",
        help="Directory to save output plots and HTML files."
    )

    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"Created output directory: {args.output}")

    # 2. Data Parsing (ETL)
    print(f"\n[Phase 1] Parsing Abstracts from {args.abstracts}...")
    if not os.path.exists(args.abstracts) or not os.path.exists(args.data):
        print("Error: Data files not found. Please check your paths.")
        return

    paper_to_authors, paper_to_text, author_to_papers = parse_abstracts(args.abstracts)

    # 3. Network Construction
    print("\n[Phase 2] Building Networks...")
    G_co, G_cit = build_networks(args.data, paper_to_authors)
    
    if not G_co or not G_cit:
        print("Error: Failed to build networks.")
        return

    # 4. Alignment & GCC Extraction
    # We focus on the intersection (authors present in both layers) for valid comparison
    print("\n[Phase 3] Aligning Network Layers...")
    common_nodes = set(G_co.nodes()) & set(G_cit.nodes())
    
    # Extract the Giant Connected Component (GCC) from the core set
    # This ensures spectral properties (eigenvalues) are well-defined
    G_co_core = G_co.subgraph(common_nodes).copy()
    largest_cc_nodes = max(nx.connected_components(G_co_core), key=len)
    
    G_social = G_co.subgraph(largest_cc_nodes).copy()
    G_intellectual = G_cit.subgraph(largest_cc_nodes).copy()

    print(f"  Final Analysis Set: {G_social.number_of_nodes()} authors (GCC of Intersection)")

    # 5. Physics & Statistical Analysis
    print("\n[Phase 4] Running Structural & Physics Analysis...")
    
    # Structural Metrics
    print_global_metrics(G_social)
    print_top_authors(G_social, G_intellectual)
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
    print("\n[Phase 5] Detecting Communities & Topics...")
    analyze_communities_robust(G_social, author_to_papers, paper_to_text, n_iterations=5)

    # 7. Visualization
    print("\n[Phase 6] Generating Interactive Map...")
    visualize_network(G_social, title=f"{args.output}/interactive_map.html")

    print(f"\nDone! All results saved to: {args.output}")

if __name__ == "__main__":
    main()
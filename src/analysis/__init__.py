"""
HEP-Th Network Analysis Package.

This package contains modules for:
1. Structural Analysis (Graph Theory basics, Centrality)
2. Physics Analysis (Spectral properties, Percolation, Power laws)
3. Community Analysis (Louvain clustering, NLP topic modeling)
"""

from .structural import (
    get_global_metrics,
    get_top_authors,
    export_centrality_tables,
    plot_top_centralities,
    analyze_layer_shortest_paths,
    analyze_strength_distribution,
    analyze_multiplex_correlation,
    analyze_degree_correlation,
)

from .physics import (
    analyze_power_law,
    analyze_spectral_properties,
    analyze_robustness,
    analyze_configuration_model,
)

from .communities import (
    check_community_distribution,
    analyze_communities_robust,
)

__all__ = [
    "get_global_metrics",
    "get_top_authors",
    "export_centrality_tables",
    "plot_top_centralities",
    "analyze_layer_shortest_paths",
    "analyze_strength_distribution",
    "analyze_multiplex_correlation",
    "analyze_degree_correlation",
    "analyze_power_law",
    "analyze_spectral_properties",
    "analyze_robustness",
    "analyze_configuration_model",
    "check_community_distribution",
    "analyze_communities_robust",
]

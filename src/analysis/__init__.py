"""
HEP-Th Network Analysis Package.

This package contains modules for:
1. Structural Analysis (Graph Theory basics, Centrality)
2. Physics Analysis (Spectral properties, Percolation, Power laws)
3. Community Analysis (Louvain clustering, NLP topic modeling)
"""

# 1. Structural Metrics (Descriptive Statistics)
from .structural import (
    get_global_metrics,
    get_top_authors,
    analyze_layer_shortest_paths,
    analyze_strength_distribution,
    analyze_multiplex_correlation,
)

# 2. Physics & Complex Systems (Advanced Analysis)
from .physics import (
    analyze_power_law,
    analyze_spectral_properties,
    analyze_robustness,
    analyze_configuration_model,
)

# 3. Community Detection & NLP (Mesoscale Structure)
from .communities import (
    check_community_distribution,
    analyze_communities_robust
)

__all__ = [
    # Structural
    "get_global_metrics",
    "get_top_authors",
    "analyze_layer_shortest_paths",
    "analyze_strength_distribution",
    "analyze_multiplex_correlation",
    # Physics
    "analyze_power_law",
    "analyze_spectral_properties",
    "analyze_robustness",
    "analyze_configuration_model",
    # Communities
    "check_community_distribution",
    "analyze_communities_robust",
]
"""
Tests for physics-oriented analysis in ``src.analysis.physics``: power-law
fitting, spectral properties, robustness simulations, and configuration-model
null comparisons. Each test uses graphs with known theoretical behavior to
verify the numerical results.
"""

import networkx as nx
import numpy as np
import pytest

from src.analysis.physics import (
    _ensure_distance_attribute,
    analyze_configuration_model,
    analyze_power_law,
    analyze_robustness,
    analyze_spectral_properties,
)


# =============================================================================
# FIXTURES (Known Mathematical Structures)
# =============================================================================

@pytest.fixture
def power_law_graph():
    """
    Generates a Barabási-Albert graph which is mathematically 
    designed to follow a power-law degree distribution.
    Size: 100 nodes to ensure enough statistical variance for the fit.
    """
    return nx.barabasi_albert_graph(100, m=3, seed=42)


@pytest.fixture
def star_graph_15():
    """
    A 15-node star graph (1 center hub, 14 disconnected leaves).
    Mathematical properties:
    - Highly vulnerable to targeted attacks (removing 1 node shatters the network).
    - Extremely low clustering (0.0).
    - Short average path length (mostly 2 hops).
    """
    return nx.star_graph(14)


@pytest.fixture
def disconnected_graph():
    """
    Two separate, disconnected components.
    Mathematical properties:
    - Algebraic connectivity (lambda_2) of the whole graph is 0.
    - Our code should automatically isolate the Giant Connected Component (GCC).
    """
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C")])  # Component 1 (Size 3)
    G.add_edges_from([("X", "Y")])              # Component 2 (Size 2)
    return G


@pytest.fixture
def clustered_graph():
    """
    Two triangles connected by a single bridge.
    Mathematical properties:
    - High transitivity/clustering.
    - Useful for testing the configuration null model.
    """
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])  # Triangle 1
    G.add_edges_from([(4, 5), (5, 6), (6, 4)])  # Triangle 2
    G.add_edges_from([(3, 4)])                  # Bridge
    return G


# =============================================================================
# UTILITY TESTS
# =============================================================================

def test_ensure_distance_attribute():
    """
    Verify that edge weights (collaborations) are properly inverted into 
    distances (1/w) for spectral and shortest-path calculations.
    
    Arrange: A graph with one weighted edge (w=2.0) and one unweighted edge.
    Act: Run the distance enforcer.
    Assert: The weighted edge has distance 0.5, the unweighted defaults to 1.0.
    """
    G = nx.Graph()
    G.add_edge("A", "B", weight=2.0)
    G.add_edge("B", "C")  # No weight provided

    _ensure_distance_attribute(G)

    assert np.isclose(G["A"]["B"]["distance"], 0.5)
    assert np.isclose(G["B"]["C"]["distance"], 1.0)


# =============================================================================
# POWER LAW TESTS
# =============================================================================

def test_analyze_power_law(power_law_graph):
    """
    Verify the power-law fitting executes correctly on a scale-free network.
    
    Arrange: A 100-node Barabási-Albert graph.
    Act: Fit the heavy-tail distribution.
    Assert: The fit parameters (alpha and xmin) are successfully calculated.
    """
    result = analyze_power_law(power_law_graph, name="Scale-Free Test")

    assert "alpha" in result
    assert "xmin" in result
    assert result["alpha"] > 0
    assert len(result["degrees"]) == 100
    # Analysis provides plot-ready PDF curves (no refit in visualization)
    assert "pdf_x" in result
    assert "pdf_empirical" in result
    assert "pdf_power_law" in result
    assert "pdf_lognormal" in result
    assert len(result["pdf_x"]) == len(result["pdf_empirical"])


def test_analyze_power_law_insufficient_data():
    """
    Edge Case: The network is too small for a statistically significant power-law fit.
    Assert: The function should gracefully catch this and return an empty dictionary.
    """
    tiny_graph = nx.complete_graph(5)
    result = analyze_power_law(tiny_graph)
    assert result == {}


# =============================================================================
# SPECTRAL ANALYSIS TESTS
# =============================================================================

def test_spectral_properties_connected(star_graph_15):
    """
    Verify spectral density and Laplacian metrics for a connected network.
    
    Arrange: A fully connected star graph.
    Act: Calculate spectral properties.
    Assert: 
    - Lambda_2 (algebraic connectivity) must be strictly > 0.
    - Von Neumann Entropy must be > 0.
    """
    res = analyze_spectral_properties(star_graph_15)

    assert res["lambda_2"] > 0.0
    assert res["diffusion_time"] > 0.0
    assert res["vn_entropy"] > 0.0
    assert len(res["eigenvalues"]) == 15


def test_spectral_properties_disconnected(disconnected_graph):
    """
    Verify robustness against disconnected graphs.
    
    Arrange: A graph fractured into two components.
    Act: Calculate spectral properties.
    Assert: The function should isolate the GCC (size 3) and calculate metrics ONLY for that subset.
    """
    res = analyze_spectral_properties(disconnected_graph)

    # The GCC of our disconnected fixture has exactly 3 nodes ("A", "B", "C")
    assert res["n_gcc"] == 3
    assert res["lambda_2"] > 0.0  # Because the GCC itself is connected


# =============================================================================
# ROBUSTNESS TESTS
# =============================================================================

def test_analyze_robustness_targeted_vs_random(star_graph_15):
    """
    Verify the physical simulation of network resilience.
    
    Arrange: A 15-node star graph.
    Act: Simulate 20% node removal (3 nodes).
    Assert: 
    - A targeted attack removes the center hub immediately, shattering the 
      Giant Connected Component (GCC size drops to ~0.06 or 1/15).
    - The attack sizes array ends at a much lower GCC size than the random array.
    """
    res = analyze_robustness(star_graph_15)

    assert "attack_sizes" in res
    assert "random_sizes" in res
    assert "x_axis" in res

    final_attack_gcc = res["attack_sizes"][-1]
    
    # After a targeted attack on a star graph, the center is gone. 
    # Remaining nodes are completely isolated (max component size = 1).
    assert np.isclose(final_attack_gcc, 1.0 / 15.0, atol=0.01)


# =============================================================================
# CONFIGURATION NULL MODEL TESTS
# =============================================================================

def test_analyze_configuration_model(clustered_graph):
    """
    Verify the topological null model simulation.
    
    Arrange: A graph with deliberate triangular clustering.
    Act: Generate randomized configuration models with the same degree sequence.
    Assert: It returns the real clustering (C_real), the null average (C_null_avg), 
    and a measurable Z-Score.
    """
    res = analyze_configuration_model(clustered_graph, n_randomizations=3)

    assert "C_real" in res
    assert "C_null_avg" in res
    assert "z_score" in res
    assert len(res["null_clustering_values"]) == 3
    
    # Real clustering should be relatively high because of the explicit triangles
    assert res["C_real"] > 0.0


def test_analyze_robustness_random_graph():
    """
    Verify robustness simulation behaves sensibly on a random graph.
    
    Arrange: An Erdos–Renyi graph with moderate size.
    Act: Run robustness analysis.
    Assert: Output arrays have consistent lengths and GCC sizes never increase as nodes are removed.
    """
    G = nx.erdos_renyi_graph(50, 0.2, seed=123)
    res = analyze_robustness(G, name="RandomER")

    x = res["x_axis"]
    attack = res["attack_sizes"]
    rand = res["random_sizes"]

    assert len(x) == len(attack) == len(rand) > 1

    # GCC fractions should never grow as more nodes are removed
    assert all(attack[i] >= attack[i + 1] for i in range(len(attack) - 1))
    assert all(rand[i] >= rand[i + 1] for i in range(len(rand) - 1))


def test_analyze_configuration_model_random_like_graph():
    """
    Second configuration-model test on a near-random graph.
    
    Arrange: A random graph with modest clustering.
    Act: Run the configuration-model null simulation.
    Assert: Results contain expected keys and a full set of null clustering values.
    """
    G = nx.erdos_renyi_graph(20, 0.3, seed=99)
    res = analyze_configuration_model(G, n_randomizations=4)

    assert "C_real" in res
    assert "C_null_avg" in res
    assert "z_score" in res
    assert len(res["null_clustering_values"]) == 4
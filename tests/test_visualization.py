"""
Tests for visualization utilities in ``src.visualization``. These tests do not
validate plot aesthetics, only that HTML/PDF outputs are produced for valid
data and that plotting functions exit gracefully when given empty inputs.
"""

import os

import networkx as nx
import numpy as np
import pytest

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


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def viz_dir(tmp_path):
    """
    Creates a temporary directory for visualization outputs.
    Pytest automatically cleans this up after the test suite finishes.
    """
    out_dir = tmp_path / "test_plots"
    out_dir.mkdir()
    return str(out_dir)


@pytest.fixture
def mock_centrality_data():
    """Mock structural data mimicking the output of export_centrality_tables."""
    return {
        "top_betweenness_data": [("Hub_A", 0.45), ("Hub_B", 0.32)],
        "top_closeness_data": [("Center_A", 0.65), ("Center_B", 0.60)],
    }


@pytest.fixture
def mock_physics_data():
    """Mock physics data mimicking heavy-tail and spectral calculations (analysis output)."""
    deg = np.array([10, 5, 5, 2, 2, 2, 1, 1, 1, 1, 1, 1])
    x_plot = np.unique(deg)
    _, counts = np.unique(deg, return_counts=True)
    pdf_empirical = np.asarray(counts, dtype=float) / len(deg)
    # Fake fit curves (decaying)
    pdf_pl = pdf_empirical * 0.9 + 0.01
    pdf_ln = pdf_empirical * 0.8 + 0.02
    return {
        "degrees": deg,
        "alpha": 2.5,
        "xmin": 2.0,
        "pdf_x": x_plot,
        "pdf_empirical": pdf_empirical,
        "pdf_power_law": pdf_pl,
        "pdf_lognormal": pdf_ln,
        "x_label": "Degree (k)",
        "eigenvalues": np.array([0.0, 0.5, 1.2, 1.5, 2.0]),
        "vn_entropy": 1.45,
    }


# =============================================================================
# INTERACTIVE NETWORK PLOT TESTS
# =============================================================================

def test_visualize_network_html_generation(viz_dir):
    """
    Verify the pyvis HTML projection.
    
    Arrange: A standard connected graph (Karate Club).
    Act: Generate the interactive topology map.
    Assert: A valid HTML file is created containing network rendering scripts.
    """
    G = nx.karate_club_graph()
    test_file = os.path.join(viz_dir, "map.html")
    
    visualize_network(G, title=test_file)

    assert os.path.exists(test_file)

    with open(test_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert "<html>" in content.lower() or "<!doctype html>" in content.lower()
        assert "nodes" in content.lower()


def test_visualize_network_large_graph_pruning(viz_dir):
    """
    Verify rendering of a large graph completes without error.
    
    Arrange: A graph with 600 nodes.
    Act: Plot the network (caller may pass partition; here we omit for backward compat).
    Assert: The HTML file is generated.
    """
    G_large = nx.barabasi_albert_graph(n=600, m=2)
    test_file = os.path.join(viz_dir, "large_map.html")
    visualize_network(G_large, title=test_file)
    assert os.path.exists(test_file)


def test_visualize_network_with_partition(viz_dir):
    """
    Verify that when the analysis layer provides a partition, visualization uses it.
    """
    from src.analysis.communities import get_community_partition

    G = nx.karate_club_graph()
    partition = get_community_partition(G, random_state=0)
    test_file = os.path.join(viz_dir, "map_with_partition.html")
    visualize_network(G, title=test_file, partition=partition)
    assert os.path.exists(test_file)
    with open(test_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert "nodes" in content.lower()


# =============================================================================
# DATA-DRIVEN PDF GENERATION TESTS
# =============================================================================

def test_plot_top_centralities(viz_dir, mock_centrality_data):
    """
    Verify that matplotlib correctly ingests dictionary data and generates the 
    horizontal bar charts for betweenness/closeness.
    """
    plot_top_centralities(mock_centrality_data, output_dir=viz_dir, top_n=2)
    
    expected_file = os.path.join(viz_dir, "centrality_betweenness_closeness_social.pdf")
    assert os.path.exists(expected_file)


def test_plot_physics_distributions(viz_dir, mock_physics_data):
    """
    Verify the generation of complex physics distributions (Power Law & Spectral Density).
    """
    # Test Power Law PDF
    plot_power_law(mock_physics_data, name="TestNet", output_dir=viz_dir)
    expected_pl = os.path.join(viz_dir, "testnet_degree_distribution_powerlaw.pdf")
    assert os.path.exists(expected_pl)

    # Test Spectral Density PDF
    plot_spectral_density(mock_physics_data, name="TestNet", output_dir=viz_dir)
    expected_spec = os.path.join(viz_dir, "testnet_spectral_density_entropy.pdf")
    assert os.path.exists(expected_spec)


def test_plot_robustness_and_correlation(viz_dir):
    """
    Verify the generation of robustness simulations and multiplex hexbin plots.
    """
    # Mock robustness data
    rob_data = {
        "x_axis": [0.0, 0.1, 0.2],
        "random_sizes": [1.0, 0.9, 0.8],
        "attack_sizes": [1.0, 0.5, 0.1],
    }
    plot_robustness(rob_data, name="TestNet", output_dir=viz_dir)
    assert os.path.exists(os.path.join(viz_dir, "testnet_network_robustness.pdf"))

    # Mock Multiplex Hexbin data
    corr_data = {
        "x_plot": np.array([0.1, 0.2, 0.3]),
        "y_plot": np.array([0.4, 0.5, 0.6]),
        "correlation": 0.85
    }
    plot_multiplex_correlation(corr_data, output_dir=viz_dir)
    assert os.path.exists(os.path.join(viz_dir, "multiplex_pagerank_vs_betweenness.pdf"))


# =============================================================================
# VALID-DATA TESTS FOR REMAINING PLOT FUNCTIONS
# =============================================================================

def test_plot_cross_layer_paths_valid_data(viz_dir):
    """With valid cross-layer path data, both hops and weighted PDFs are generated."""
    data = {
        "distances_hops": [1, 2, 2, 3],
        "distances_weighted": [1.0, 2.0, 2.5, 3.0],
        "avg_hops": 2.0,
        "avg_weighted": 2.125,
    }
    plot_cross_layer_paths(data, output_dir=viz_dir)
    assert os.path.exists(os.path.join(viz_dir, "cross_layer_path_distribution_hops.pdf"))
    assert os.path.exists(os.path.join(viz_dir, "cross_layer_path_distribution_weighted.pdf"))


def test_plot_strength_distribution_valid_data(viz_dir):
    """With valid strength/degree arrays, the correlation PDF is generated."""
    data = {
        "k_values": np.array([1, 1, 2, 2, 3]),
        "s_values": np.array([2.0, 3.0, 6.0, 8.0, 12.0]),
        "k_unique": np.array([1.0, 2.0, 3.0]),
        "s_avg_k": np.array([2.5, 7.0, 12.0]),
        "beta": 1.2,
        "intercept": 0.1,
    }
    plot_strength_distribution(data, name="TestLayer", output_dir=viz_dir)
    assert os.path.exists(os.path.join(viz_dir, "testlayer_strength_degree_correlation.pdf"))


def test_plot_degree_vs_instrength_valid_data(viz_dir):
    """With valid x_plot/y_plot, the degree vs in-strength PDF is generated."""
    data = {
        "x_plot": np.array([1.0, 2.0, 3.0, 5.0]),
        "y_plot": np.array([10.0, 50.0, 100.0, 200.0]),
        "correlation": 0.92,
    }
    plot_degree_vs_instrength(data, output_dir=viz_dir)
    assert os.path.exists(os.path.join(viz_dir, "degree_vs_instrength_social_vs_citation.pdf"))


def test_plot_configuration_model_valid_data(viz_dir):
    """With null model clustering values, the configuration-model PDF is generated."""
    data = {
        "null_clustering_values": [0.3, 0.35, 0.32, 0.28, 0.4],
        "C_real": 0.45,
        "z_score": 2.1,
    }
    plot_configuration_model(data, name="TestNet", output_dir=viz_dir)
    assert os.path.exists(os.path.join(viz_dir, "testnet_configuration_model_clustering.pdf"))


def test_plot_community_distribution_valid_data(viz_dir):
    """With community sizes list, the community size distribution PDF is generated."""
    data = {"sizes": [50, 30, 20, 10, 5, 5, 2, 2, 2]}
    plot_community_distribution(data, layer_name="TestLayer", output_dir=viz_dir)
    assert os.path.exists(os.path.join(viz_dir, "testlayer_community_size_distribution.pdf"))


# =============================================================================
# EDGE CASE: EMPTY DATA HANDLING
# =============================================================================

@pytest.mark.parametrize(
    "plot_function, args",
    [
        (plot_power_law, {"data": {}, "name": "Empty", "output_dir": "should_not_exist"}),
        (plot_top_centralities, {"data": {}, "output_dir": "should_not_exist"}),
        (plot_community_distribution, {"data": {}, "layer_name": "Empty", "output_dir": "should_not_exist"}),
    ],
)
def test_empty_data_graceful_exit(plot_function, args, tmp_path):
    """
    Verify that plotting functions gracefully abort when passed empty data dictionaries.
    
    Act: Call various plotting functions with an empty dictionary.
    Assert: They should return immediately without crashing and without creating the output directory.
    """
    # Update the args with a safe temp path just in case it fails and tries to write
    bad_dir = tmp_path / "bad_dir"
    args["output_dir"] = str(bad_dir)
    
    plot_function(**args)
    
    # The directory should not have been created because the function aborted early
    assert not os.path.exists(str(bad_dir))
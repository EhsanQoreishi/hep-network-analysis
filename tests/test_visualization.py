"""
Tests for visualization utilities in ``src.visualization``. These tests do not
validate plot aesthetics, only that HTML/PDF outputs are produced for valid
data and that plotting functions exit gracefully when given empty inputs.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

from src.visualization import (
    _savefig,
    _sanitize_name,
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
# HELPER TESTS (_sanitize_name, _savefig)
# =============================================================================

@pytest.mark.parametrize(
    "name, expected",
    [
        ("Social Layer", "social_layer"),
        ("Citation-Layer", "citation_layer"),
        ("Test/Path", "test_path"),
        ("A (B)", "a_b"),
        ("X: Y", "x_y"),
        ("  spaces  ", "spaces"),
        ("double__underscore", "double_underscore"),
    ],
)
def test_sanitize_name(name, expected):
    """Verify layer/network names are turned into safe filename-safe strings (no spaces, slashes, parens)."""
    assert _sanitize_name(name) == expected


def test_savefig(tmp_path):
    """Verify _savefig writes a non-empty PDF from the current figure."""
    path = str(tmp_path / "out")
    plt.figure()
    plt.plot([1, 2], [1, 2])
    _savefig(path)
    plt.close()
    pdf_path = tmp_path / "out.pdf"
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 100


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


def test_plot_top_centralities_respects_top_n(viz_dir):
    """With top_n=1, only one bar per panel is plotted; PDF is still generated."""
    data = {
        "top_betweenness_data": [("A", 0.5), ("B", 0.3)],
        "top_closeness_data": [("A", 0.6), ("B", 0.4)],
    }
    plot_top_centralities(data, output_dir=viz_dir, top_n=1)
    assert os.path.exists(os.path.join(viz_dir, "centrality_betweenness_closeness_social.pdf"))


def test_plot_physics_distributions(viz_dir, mock_physics_data):
    """
    Verify the generation of complex physics distributions (Power Law & Spectral Density).
    """
    # Test Power Law PDF (fallback path: precomputed arrays, no fit object)
    plot_power_law(mock_physics_data, name="TestNet", output_dir=viz_dir)
    expected_pl = os.path.join(viz_dir, "testnet_degree_distribution_powerlaw.pdf")
    assert os.path.exists(expected_pl)

    # Test Spectral Density PDF
    plot_spectral_density(mock_physics_data, name="TestNet", output_dir=viz_dir)
    expected_spec = os.path.join(viz_dir, "testnet_spectral_density_entropy.pdf")
    assert os.path.exists(expected_spec)


def test_plot_spectral_density_few_eigenvalues(viz_dir):
    """Spectral density with few eigenvalues uses hist when KDE is not applicable; PDF is still generated."""
    data = {"eigenvalues": np.array([0.0, 0.5, 1.0]), "vn_entropy": 0.8}
    plot_spectral_density(data, name="Tiny", output_dir=viz_dir)
    assert os.path.exists(os.path.join(viz_dir, "tiny_spectral_density_entropy.pdf"))


# =============================================================================
# PLOT_POWER_LAW BEHAVIOR: FIT PATH VS FALLBACK PATH
# =============================================================================

def test_plot_power_law_uses_fit_path_when_fit_present(viz_dir):
    """
    When data contains a powerlaw Fit object (from analyze_power_law), the
    visualization uses the library's plot_pdf and produces the degree-distribution PDF.
    """
    from src.analysis.physics import analyze_power_law

    G = nx.barabasi_albert_graph(80, m=2, seed=123)
    result = analyze_power_law(G, name="BA")
    assert "fit" in result
    plot_power_law(result, name="BA", output_dir=viz_dir)
    expected = os.path.join(viz_dir, "ba_degree_distribution_powerlaw.pdf")
    assert os.path.exists(expected)
    assert os.path.getsize(expected) > 500


def test_plot_power_law_uses_fallback_path_when_no_fit(viz_dir, mock_physics_data):
    """
    When data has no 'fit' key but has pdf_x and pdf_empirical, the fallback
    (precomputed arrays) path is used and still produces a valid PDF.
    """
    assert "fit" not in mock_physics_data
    plot_power_law(mock_physics_data, name="FallbackNet", output_dir=viz_dir)
    expected = os.path.join(viz_dir, "fallbacknet_degree_distribution_powerlaw.pdf")
    assert os.path.exists(expected)
    assert os.path.getsize(expected) > 200


def test_plot_power_law_empty_or_insufficient_data_creates_no_file(tmp_path):
    """
    When data is empty or pdf_x has fewer than 2 points, no PDF is written.
    """
    out = str(tmp_path / "out")
    plot_power_law({}, name="Empty", output_dir=out)
    assert not os.path.exists(os.path.join(out, "empty_degree_distribution_powerlaw.pdf"))

    plot_power_law({"pdf_x": np.array([1.0]), "pdf_empirical": np.array([1.0])}, name="One", output_dir=out)
    assert not os.path.exists(os.path.join(out, "one_degree_distribution_powerlaw.pdf"))


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


def test_plot_robustness_many_fractions(viz_dir):
    """Robustness plot with many removal fractions produces a smooth curve PDF."""
    rob_data = {
        "x_axis": [i * 0.05 for i in range(21)],
        "random_sizes": [1.0 - i * 0.04 for i in range(21)],
        "attack_sizes": [1.0] + [0.9 - i * 0.04 for i in range(1, 21)],
    }
    plot_robustness(rob_data, name="Smooth", output_dir=viz_dir)
    assert os.path.exists(os.path.join(viz_dir, "smooth_network_robustness.pdf"))

    # Mock Multiplex Hexbin data
    corr_data = {
        "x_plot": np.array([0.1, 0.2, 0.3]),
        "y_plot": np.array([0.4, 0.5, 0.6]),
        "correlation": 0.85,
    }
    plot_multiplex_correlation(corr_data, output_dir=viz_dir)
    assert os.path.exists(os.path.join(viz_dir, "multiplex_pagerank_vs_betweenness.pdf"))


def test_plot_multiplex_correlation_negative_correlation(viz_dir):
    """Multiplex correlation plot is generated with negative correlation value in title."""
    data = {
        "x_plot": np.array([0.5, 1.0, 2.0]),
        "y_plot": np.array([2.0, 1.0, 0.5]),
        "correlation": -0.7,
    }
    plot_multiplex_correlation(data, output_dir=viz_dir)
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


def test_plot_cross_layer_paths_hops_only(viz_dir):
    """When only distances_hops is provided, both PDFs are still written (weighted uses empty hist)."""
    data = {"distances_hops": [1, 1, 2], "distances_weighted": [], "avg_hops": 1.33, "avg_weighted": 0.0}
    plot_cross_layer_paths(data, output_dir=viz_dir)
    assert os.path.exists(os.path.join(viz_dir, "cross_layer_path_distribution_hops.pdf"))
    assert os.path.exists(os.path.join(viz_dir, "cross_layer_path_distribution_weighted.pdf"))


def test_plot_cross_layer_paths_weighted_only(viz_dir):
    """When only distances_weighted is provided, both PDFs are still written (hops uses empty hist)."""
    data = {"distances_hops": [], "distances_weighted": [1.0, 2.0, 2.0], "avg_hops": 0.0, "avg_weighted": 1.67}
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


def test_plot_strength_distribution_citation_layer(viz_dir):
    """Strength distribution with Citation-Layer name uses in-degree/in-strength labels and correct filename."""
    data = {
        "k_values": np.array([2, 3, 4]),
        "s_values": np.array([10.0, 20.0, 40.0]),
        "k_unique": np.array([2.0, 3.0, 4.0]),
        "s_avg_k": np.array([10.0, 20.0, 40.0]),
        "beta": 1.5,
        "intercept": 0.0,
    }
    plot_strength_distribution(data, name="Citation-Layer", output_dir=viz_dir)
    assert os.path.exists(os.path.join(viz_dir, "citation_layer_strength_degree_correlation.pdf"))


def test_plot_degree_vs_instrength_valid_data(viz_dir):
    """With valid x_plot/y_plot, the degree vs in-strength PDF is generated."""
    data = {
        "x_plot": np.array([1.0, 2.0, 3.0, 5.0]),
        "y_plot": np.array([10.0, 50.0, 100.0, 200.0]),
        "correlation": 0.92,
    }
    plot_degree_vs_instrength(data, output_dir=viz_dir)
    assert os.path.exists(os.path.join(viz_dir, "degree_vs_instrength_social_vs_citation.pdf"))


def test_plot_degree_vs_instrength_sparse_data(viz_dir):
    """Hexbin plot is generated even with few points (two pairs)."""
    data = {
        "x_plot": np.array([1.0, 2.0]),
        "y_plot": np.array([5.0, 25.0]),
        "correlation": 1.0,
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


def test_plot_configuration_model_negative_z_score(viz_dir):
    """Configuration model plot with negative Z-score (real below null) still produces PDF."""
    data = {
        "null_clustering_values": [0.4, 0.42, 0.38],
        "C_real": 0.2,
        "z_score": -1.5,
    }
    plot_configuration_model(data, name="BelowNull", output_dir=viz_dir)
    assert os.path.exists(os.path.join(viz_dir, "belownull_configuration_model_clustering.pdf"))


def test_plot_community_distribution_valid_data(viz_dir):
    """With community sizes list, the community size distribution PDF is generated."""
    data = {"sizes": [50, 30, 20, 10, 5, 5, 2, 2, 2]}
    plot_community_distribution(data, layer_name="TestLayer", output_dir=viz_dir)
    assert os.path.exists(os.path.join(viz_dir, "testlayer_community_size_distribution.pdf"))


def test_plot_community_distribution_few_communities(viz_dir):
    """Community distribution with few large communities still produces valid PDF."""
    data = {"sizes": [100, 50, 25]}
    plot_community_distribution(data, layer_name="Citation-Layer", output_dir=viz_dir)
    assert os.path.exists(os.path.join(viz_dir, "citation_layer_community_size_distribution.pdf"))


# =============================================================================
# EDGE CASE: EMPTY DATA HANDLING (all plot functions)
# =============================================================================

@pytest.mark.parametrize(
    "plot_function, args",
    [
        (plot_power_law, {"data": {}, "name": "Empty"}),
        (plot_top_centralities, {"data": {}}),
        (plot_community_distribution, {"data": {}, "layer_name": "Empty"}),
        (plot_cross_layer_paths, {"data": {}}),
        (plot_strength_distribution, {"data": {}, "name": "Empty"}),
        (plot_multiplex_correlation, {"data": {}}),
        (plot_degree_vs_instrength, {"data": {}}),
        (plot_spectral_density, {"data": {}, "name": "Empty"}),
        (plot_robustness, {"data": {}, "name": "Empty"}),
        (plot_configuration_model, {"data": {}, "name": "Empty"}),
    ],
)
def test_empty_data_graceful_exit(plot_function, args, tmp_path):
    """
    Verify that every plotting function gracefully aborts when passed empty or
    missing data and does not create the output directory.
    """
    bad_dir = tmp_path / "bad_dir"
    args["output_dir"] = str(bad_dir)
    plot_function(**args)
    assert not os.path.exists(str(bad_dir))
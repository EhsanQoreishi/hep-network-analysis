import networkx as nx
import numpy as np
import pytest

from src.analysis.communities import analyze_communities_robust
from src.analysis.physics import analyze_power_law, analyze_spectral_properties
from src.analysis.structural import analyze_layer_shortest_paths, get_global_metrics


@pytest.fixture
def karate_graph():
    """
    Standard benchmark graph (Zachary's Karate Club).
    Small, connected, and well-understood properties.
    """
    return nx.karate_club_graph()


@pytest.fixture
def temp_results(tmp_path):
    """Automatic temporary directory for output plots."""
    return tmp_path


# --- Structural Tests ---


def test_structural_metrics(karate_graph):
    """Verify standard topological metrics calculation."""
    metrics = get_global_metrics(karate_graph)

    assert metrics["nodes"] == 34
    assert metrics["edges"] == 78
    assert 0 <= metrics["density"] <= 1
    assert 0 <= metrics["transitivity"] <= 1


# --- Physics Tests ---


def test_power_law_analysis(karate_graph, temp_results):
    """
    Test Power Law fitting.
    Note: Karate club is too small for a valid power law, but the code should run without crashing.
    """
    results = analyze_power_law(
        karate_graph, name="TestGraph", output_dir=str(temp_results)
    )

    assert "alpha" in results
    assert "xmin" in results
    assert (temp_results / "testgraph_power_law_fit.png").exists()


def test_spectral_properties(karate_graph, temp_results):
    """
    Test Laplacian spectral properties (Entropy & Connectivity).
    """
    metrics = analyze_spectral_properties(karate_graph, output_dir=str(temp_results))
    assert metrics["lambda_2"] > 0
    assert np.isclose(metrics["diffusion_time"], 1.0 / metrics["lambda_2"])
    assert metrics["vn_entropy"] > 0


# --- Community Detection Tests ---


def test_communities_robust(karate_graph):
    """
    Test Louvain community detection with text integration.
    """
    nodes = [str(n) for n in karate_graph.nodes()]
    mock_auth_to_papers = {n: ["P1"] for n in nodes}
    mock_paper_to_text = {"P1": "quantum physics theory model"}

    partition = analyze_communities_robust(
        karate_graph, mock_auth_to_papers, mock_paper_to_text, n_iterations=2
    )

    assert isinstance(partition, dict)
    assert len(partition) == len(karate_graph.nodes())
    assert len(set(partition.values())) > 0


def test_cross_layer_path(karate_graph, temp_results):
    """
    Test distance calculation between layers.
    We simulate a directed citation layer using the same structure.
    """
    G_cit = nx.to_directed(karate_graph)

    avg_dist = analyze_layer_shortest_paths(
        G_cit, karate_graph, output_dir=str(temp_results)
    )

    assert isinstance(avg_dist, float)
    assert avg_dist >= 1.0

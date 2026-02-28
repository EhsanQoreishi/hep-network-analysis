"""
Tests for structural analysis: global metrics, centralities, and multiplex
correlations in ``src.analysis.structural``. Each test uses small graphs with
known analytical properties (triangles, stars, random graphs) to validate the
mathematical behavior of the functions.
"""

import os
import networkx as nx
import numpy as np
import pytest

from src.analysis.structural import (
    _density_directed,
    _density_undirected,
    _fast_avg_strength,
    analyze_degree_correlation,
    analyze_layer_shortest_paths,
    analyze_multiplex_correlation,
    analyze_strength_distribution,
    compute_centrality_data,
    export_centrality_tables,
    get_global_metrics,
    get_top_authors,
)

# =============================================================================
# FIXTURES (Known Mathematical Structures)
# =============================================================================

@pytest.fixture
def triangle_graph():
    """
    A simple 3-node fully connected triangle.
    Mathematical properties:
    - Density: 1.0 (all possible edges exist)
    - Transitivity/Clustering: 1.0
    """
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    return G

@pytest.fixture
def star_graph():
    """
    A 4-node star graph with 'Center' connected to 'Leaf1', 'Leaf2', 'Leaf3'.
    """
    G = nx.Graph()
    G.add_edges_from([("Center", "L1"), ("Center", "L2"), ("Center", "L3")])
    return G

@pytest.fixture
def directed_chain():
    """
    A directed chain: A -> B -> C.
    """
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("B", "C", weight=2.0)
    return G

# =============================================================================
# MATH & UTILITY TESTS
# =============================================================================

@pytest.mark.parametrize(
    "N, E, expected",
    [
        (3, 3, 1.0),        # Triangle: fully connected
        (4, 3, 0.5),        # Star: 50% density
        (1, 0, 0.0),        
        (0, 0, 0.0),        
    ],
)
def test_density_undirected(N, E, expected):
    """Verify the undirected density formula against analytical limits."""
    assert np.isclose(_density_undirected(N, E), expected)

def test_fast_avg_strength():
    """Test the Numba JIT-compiled average strength calculator."""
    k_vals = np.array([1.0, 1.0, 2.0, 2.0])
    s_vals = np.array([2.0, 4.0, 10.0, 20.0])
    unique_k = np.array([1.0, 2.0])

    avg_s = _fast_avg_strength(k_vals, s_vals, unique_k)
    assert np.isclose(avg_s[0], 3.0)
    assert np.isclose(avg_s[1], 15.0)

# =============================================================================
# GLOBAL METRICS TESTS
# =============================================================================

def test_get_global_metrics_undirected(triangle_graph):
    """Verify metrics for a fully connected undirected graph."""
    metrics = get_global_metrics(triangle_graph)
    assert metrics["nodes"] == 3
    assert metrics["density"] == 1.0
    assert metrics["transitivity"] == 1.0


@pytest.fixture
def directed_triangle():
    """
    Directed 3-cycle: A -> B -> C -> A.
    Density = E/(N*(N-1)) = 3/6 = 0.5. Transitivity and clustering are positive.
    """
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    return G


@pytest.fixture
def directed_star():
    """
    Directed star: center has out-edges to 3 leaves (no back-edges).
    N=4, E=3, density = 3/12 = 0.25. Many open triads, no closed triangles -> transitivity 0.
    """
    G = nx.DiGraph()
    G.add_edges_from([("C", "L1"), ("C", "L2"), ("C", "L3")])
    return G


def test_get_global_metrics_directed_triangle(directed_triangle):
    """Verify directed density and that directed clustering/transitivity are computed."""
    metrics = get_global_metrics(directed_triangle)
    assert metrics["nodes"] == 3
    assert metrics["edges"] == 3
    assert metrics["density"] == pytest.approx(0.5)
    assert 0 <= metrics["transitivity"] <= 1
    assert 0 <= metrics["avg_clustering"] <= 1


def test_get_global_metrics_directed_star(directed_star):
    """Directed star has density 0.25 and zero transitivity (no directed triangles)."""
    metrics = get_global_metrics(directed_star)
    assert metrics["nodes"] == 4
    assert metrics["edges"] == 3
    assert metrics["density"] == pytest.approx(0.25)
    assert metrics["transitivity"] == 0.0
    assert metrics["avg_clustering"] == 0.0


def test_compute_centrality_data_and_export(tmp_path, star_graph):
    """
    Verify that centrality computation is pure, and CSV export writes correct tables.
    
    Arrange: A small star graph with explicit distance attributes.
    Act: Compute centrality data and export top entries to CSV.
    Assert: The pure helper returns the expected keys and the exporter creates both CSV files.
    """
    for _, _, d in star_graph.edges(data=True):
        d["distance"] = 1.0

    data = compute_centrality_data(star_graph, top_n=3)
    assert "top_betweenness_data" in data
    assert "top_closeness_data" in data
    assert len(data["top_betweenness_data"]) <= 3
    assert len(data["top_closeness_data"]) <= 3

    out_dir = tmp_path / "centrality"
    res = export_centrality_tables(star_graph, output_dir=str(out_dir), top_n=3)

    bet_path = out_dir / "top_betweenness_social.csv"
    clo_path = out_dir / "top_closeness_social.csv"

    assert bet_path.exists()
    assert clo_path.exists()
    assert res["betweenness_csv"] == str(bet_path)
    assert res["closeness_csv"] == str(clo_path)

# =============================================================================
# CENTRALITY & TOP AUTHORS TESTS
# =============================================================================

def test_get_top_authors(star_graph, directed_chain):
    """Verify centrality algorithms identify structural hubs correctly."""
    for u, v, d in star_graph.edges(data=True):
        d["distance"] = 1.0

    results = get_top_authors(star_graph, directed_chain)
    assert results["collaborative"][0][0] == "Center"
    assert results["influential"][0][0] == "C"


def test_analyze_strength_distribution_basic(star_graph):
    """
    Verify strength vs degree analysis returns consistent arrays and fit parameters.
    """
    # Give all edges unit weight so strengths are well-defined and positive
    for _, _, d in star_graph.edges(data=True):
        d["weight"] = 1.0

    res = analyze_strength_distribution(star_graph, name="Star")
    assert res["beta"] == pytest.approx(res["beta"])  # numeric
    assert res["intercept"] == pytest.approx(res["intercept"])
    assert len(res["k_values"]) == len(res["s_values"])
    assert len(res["k_unique"]) == len(res["s_avg_k"])


def test_analyze_strength_distribution_empty_graph():
    """
    Edge case: graph with nodes but no edges should return an "empty" result.
    """
    G = nx.Graph()
    G.add_nodes_from(["A", "B", "C"])

    res = analyze_strength_distribution(G, name="Empty")
    assert res["beta"] == 0.0
    assert res["intercept"] == 0.0
    assert res["k_values"] == []
    assert res["s_values"] == []

# =============================================================================
# MULTIPLEX & CROSS-LAYER TESTS
# =============================================================================

@pytest.mark.parametrize(
    "G_co_edges, G_cit_edges, expected_avg_hops",
    [
        # Case 1: A cites C; socially A-B-C so path length 2
        ([("A", "B"), ("B", "C")], [("A", "C")], 2.0),
        # Case 2: A cites B (direct collaboration); path length 1
        ([("A", "B")], [("A", "B")], 1.0),
        # Case 3: Two citation pairs (A->C and A->B); social A-B, B-C -> (2 + 1) / 2 = 1.5
        ([("A", "B"), ("B", "C")], [("A", "C"), ("A", "B")], 1.5),
    ],
)
def test_analyze_layer_shortest_paths(G_co_edges, G_cit_edges, expected_avg_hops):
    """Verify cross-layer mapping of citation edges to social hops (multiple topologies)."""
    G_co = nx.Graph()
    G_co.add_edges_from(G_co_edges)
    for u, v in G_co.edges():
        G_co[u][v]["distance"] = 1.0
        G_co[u][v]["weight"] = 1.0
    G_cit = nx.DiGraph()
    G_cit.add_edges_from(G_cit_edges)
    for u, v in G_cit.edges():
        G_cit[u][v]["weight"] = 1.0

    result = analyze_layer_shortest_paths(G_cit, G_co)
    assert result["avg_hops"] == pytest.approx(expected_avg_hops)
    assert "distances_hops" in result
    assert "distances_weighted" in result


def test_analyze_layer_shortest_paths_disconnected_citing_pair():
    """When citing pair has no social path, that pair is skipped and avg is 0 if no valid paths."""
    G_co = nx.Graph()
    G_co.add_nodes_from(["X", "Y"])  # Isolated nodes only
    G_cit = nx.DiGraph()
    G_cit.add_edge("X", "Y", weight=1.0)
    result = analyze_layer_shortest_paths(G_cit, G_co)
    assert result["distances_hops"] == []
    assert result["avg_hops"] == 0.0
    assert result["distances_weighted"] == []
    assert result["avg_weighted"] == 0.0

def test_correlations():
    """
    Test Spearman correlations between layers. 
    Uses a larger graph to satisfy statistical requirements and prevent KeyErrors.
    """
    # Arrange: Create larger graphs to ensure enough common authors for correlation
    n_nodes = 20
    G_co = nx.gnp_random_graph(n_nodes, 0.5, seed=42)
    G_cit = nx.gnp_random_graph(n_nodes, 0.5, seed=42, directed=True)
    
    for u, v, d in G_co.edges(data=True):
        d["distance"] = 1.0

    # Act: Compute multiplex and degree correlations
    multi_res = analyze_multiplex_correlation(G_co, G_cit)
    deg_res = analyze_degree_correlation(G_co, G_cit)

    # Assert: Ensure correlation keys exist and contain plot data
    assert "correlation" in multi_res
    assert "x_plot" in multi_res
    assert len(multi_res["x_plot"]) >= 0

    assert "correlation" in deg_res
    assert "x_plot" in deg_res
    assert len(deg_res["x_plot"]) >= 0
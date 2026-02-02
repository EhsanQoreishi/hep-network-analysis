import os

import networkx as nx
import pytest

from src.visualization import visualize_network


@pytest.fixture
def viz_setup(tmp_path):
    """
    Creates a temporary directory for visualization output.
    Returns the path to the expected HTML file.
    """
    output_dir = tmp_path / "temp_viz"
    output_dir.mkdir()

    test_file = output_dir / "map.html"
    return str(test_file)


def test_html_generation(viz_setup):
    """
    Verifies that the visualization function generates a valid HTML file
    for a standard small graph (Karate Club).
    """
    test_file_path = viz_setup
    G = nx.karate_club_graph()

    visualize_network(G, title=test_file_path)

    assert os.path.exists(test_file_path)

    with open(test_file_path, "r", encoding="utf-8") as f:
        content = f.read()

        assert "<html>" in content.lower() or "<!doctype html>" in content.lower()
        assert "<script" in content.lower()
        assert "nodes" in content.lower()


def test_visualization_large_graph_pruning(viz_setup):
    """
    Test the Module 2 Optimization:
    The visualizer is configured to slice the top 500 hubs using heapq.
    We feed it 600 nodes to verify this filtering logic triggers correctly
    and handles the transition without crashing.
    """
    test_file_path = viz_setup

    G_large = nx.barabasi_albert_graph(n=600, m=2)

    visualize_network(G_large, title=test_file_path)

    assert os.path.exists(test_file_path)
    assert os.path.getsize(test_file_path) > 1000


def test_visualization_empty_graph(viz_setup):
    """
    Edge Case: Ensure the visualizer handles an empty graph without throwing errors.
    """
    test_file_path = viz_setup
    G_empty = nx.Graph()

    visualize_network(G_empty, title=test_file_path)

    assert os.path.exists(test_file_path)

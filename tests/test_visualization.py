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
    Verifies that the visualization function generates a valid HTML file.
    """
    test_file_path = viz_setup

    G = nx.karate_club_graph()

    visualize_network(G, title=test_file_path)

    assert os.path.exists(test_file_path)

    with open(test_file_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "<html>" in content.lower() or "<!doctype html>" in content.lower()
        assert "<script" in content.lower()

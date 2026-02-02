import networkx as nx
import pytest

from src.networks import build_networks


@pytest.fixture
def network_setup(tmp_path):
    """
    Creates a dummy citation file and author mapping for testing.
    tmp_path is cleaned up automatically by pytest.
    """
    edges_content = "P1 P2\nP2 P3\nP2 P2\nP4 P2\n"

    edges_file = tmp_path / "fake_edges.txt"
    edges_file.write_text(edges_content, encoding="utf-8")

    mock_p2a = {
        "P1": ["A. User"],
        "P2": ["B. User"],
        "P3": ["A. User", "C. User"],
        "P4": ["A. User"],
    }

    return str(edges_file), mock_p2a


def test_build_networks_logic(network_setup):
    """
    Verifies that co-authorship and citation logic correctly transforms
    raw data into NetworkX graphs.
    """
    edges_file, mock_p2a = network_setup

    G_co, G_cit = build_networks(edges_file, mock_p2a)

    assert "A. User" in G_co.nodes()
    assert G_co.has_edge("A. User", "C. User")
    assert not G_co.has_edge("A. User", "B. User")
    assert G_cit.has_edge("A. User", "B. User")
    assert G_cit.has_edge("B. User", "A. User")
    assert G_cit.has_edge("B. User", "C. User")


def test_citation_weights(network_setup):
    """
    Verifies that multiple citations between the same authors increase edge weight.
    """
    edges_file, mock_p2a = network_setup
    _, G_cit = build_networks(edges_file, mock_p2a)

    weight = G_cit["A. User"]["B. User"]["weight"]
    assert weight == 2


def test_self_citation_filtering(network_setup):
    """
    Verifies that self-citations (Author X cites Author X) are removed.
    """
    edges_file, mock_p2a = network_setup
    _, G_cit = build_networks(edges_file, mock_p2a)

    if "B. User" in G_cit:
        assert not G_cit.has_edge("B. User", "B. User")


def test_empty_or_disjoint_input(tmp_path):
    """
    Ensure code doesn't crash on empty or disconnected input.
    """
    edges_file = tmp_path / "empty_edges.txt"
    edges_file.write_text("", encoding="utf-8")

    mock_p2a = {"P1": ["Lone Wolf"]}

    G_co, G_cit = build_networks(str(edges_file), mock_p2a)

    assert "Lone Wolf" in G_co.nodes()
    assert G_co.number_of_edges() == 0
    assert G_cit.number_of_edges() == 0

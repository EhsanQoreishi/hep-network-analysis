import pytest

from src.networks import build_networks


@pytest.fixture
def network_setup(tmp_path):
    """
    Creates a dummy citation file and author mapping for testing.
    tmp_path is cleaned up automatically by pytest.
    """
    edges_content = "P1 P2\nP2 P3\n"
    edges_file = tmp_path / "fake_edges.txt"
    edges_file.write_text(edges_content, encoding="utf-8")

    mock_p2a = {
        "P1": ["A. User"],
        "P2": ["B. User"],
        "P3": ["A. User", "C. User"],
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

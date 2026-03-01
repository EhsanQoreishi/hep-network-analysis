"""
Tests for network construction in ``src.networks``: bipartite projection for
co-authorship graphs and vectorized author-level citation mapping. The fixtures
encode small paper/author datasets with known edges to check weights, distances,
and self-citation removal.
"""

import networkx as nx
import pytest

from src.networks import build_networks


# =============================================================================
# FIXTURES (Known Data Structures)
# =============================================================================

@pytest.fixture
def mock_network_data(tmp_path):
    """
    Creates a highly controlled mock dataset of papers, authors, and citations.
    
    Data Structure:
    - P1: Authors A, B
    - P2: Authors B, C
    - P3: Authors A, B (A and B collaborate again!)
    - P4: Author A (Solo paper)
    
    Citations:
    - P1 cites P2 (A and B cite B and C)
    - P4 cites P1 (A cites A and B) -> Tests self-citation removal!
    - P4 cites P2 (A cites B and C)
    """
    # Create the raw citation edges file
    edges_content = "P1 P2\nP4 P1\nP4 P2\n"
    edges_file = tmp_path / "controlled_edges.txt"
    edges_file.write_text(edges_content, encoding="utf-8")

    # Create the dictionary of parsed authors
    mock_p2a = {
        "P1": ["Author A", "Author B"],
        "P2": ["Author B", "Author C"],
        "P3": ["Author A", "Author B"],
        "P4": ["Author A"],
    }

    return str(edges_file), mock_p2a


# =============================================================================
# CO-AUTHORSHIP TESTS (Bipartite Projection)
# =============================================================================

def test_coauthorship_bipartite_projection(mock_network_data):
    """
    Verify the sparse matrix multiplication (C = B * B^T) correctly accumulates weights.
    
    Arrange: A dataset where A and B co-author twice (P1, P3), B and C co-author once (P2).
    Act: Build the networks.
    Assert:
    - The edge (A, B) must exist with weight = 2.
    - The edge (B, C) must exist with weight = 1.
    - Distance must be precisely 1.0 / weight.
    - P4 (solo paper) should not create any self-loop edges for Author A.
    """
    edges_file, p2a = mock_network_data
    G_co, _ = build_networks(edges_file, p2a)

    assert G_co.has_node("Author A")
    assert G_co.has_node("Author B")
    assert G_co.has_node("Author C")

    # A and B collaborated twice (P1 and P3)
    assert G_co.has_edge("Author A", "Author B")
    assert G_co["Author A"]["Author B"]["weight"] == 2
    assert G_co["Author A"]["Author B"]["distance"] == 0.5

    # B and C collaborated once (P2)
    assert G_co.has_edge("Author B", "Author C")
    assert G_co["Author B"]["Author C"]["weight"] == 1
    assert G_co["Author B"]["Author C"]["distance"] == 1.0

    # No self-loops allowed (Author A wrote P4 alone)
    assert not G_co.has_edge("Author A", "Author A")


# =============================================================================
# CITATION TESTS (Vectorized Joins & Filtering)
# =============================================================================

def test_citation_vectorized_mapping(mock_network_data):
    """
    Verify that Paper->Paper citations correctly unfold into Author->Author citations.
    
    Arrange: P4 (Author A) cites P2 (Authors B, C).
    Act: Build the networks.
    Assert: The directed graph must contain edges (A -> B) and (A -> C).
    """
    edges_file, p2a = mock_network_data
    _, G_cit = build_networks(edges_file, p2a)

    assert G_cit.has_edge("Author A", "Author B")
    assert G_cit.has_edge("Author A", "Author C")


def test_self_citation_removal(mock_network_data):
    """
    Verify the strict removal of intellectual self-citations.
    
    Arrange: 
    - P1 (A, B) cites P2 (B, C). B citing B is a self-citation.
    - P4 (A) cites P1 (A, B). A citing A is a self-citation.
    Act: Build the networks.
    Assert: 
    - B -> B and A -> A MUST NOT exist.
    - A -> B (from P4->P1) and A -> C (from P1->P2) must still exist.
    """
    edges_file, p2a = mock_network_data
    _, G_cit = build_networks(edges_file, p2a)

    # Self-citations must be stripped
    assert not G_cit.has_edge("Author B", "Author B")
    assert not G_cit.has_edge("Author A", "Author A")

    # Valid external citations from the same papers must be preserved
    # From P1->P2: A cites B, A cites C
    assert G_cit.has_edge("Author A", "Author B")
    assert G_cit.has_edge("Author A", "Author C")

    # From P4->P1: A cites B (since A cites A is removed)
    assert G_cit.has_edge("Author A", "Author B")


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

def test_build_networks_empty_data(tmp_path):
    """
    Verify graceful handling of disconnected or entirely empty datasets.
    
    Arrange: An empty edge file and an author dict with only solo papers.
    Act: Build the networks.
    Assert: Graphs are returned without crashing, containing nodes but zero edges.
    """
    empty_file = tmp_path / "empty_edges.txt"
    empty_file.write_text("", encoding="utf-8")

    mock_p2a = {"P1": ["Lone Wolf"], "P2": ["Independent Researcher"]}

    G_co, G_cit = build_networks(str(empty_file), mock_p2a)

    # Nodes exist but no connections were made
    assert G_co.number_of_nodes() == 2
    assert G_co.number_of_edges() == 0
    assert G_cit.number_of_edges() == 0


def test_citation_edge_weights_aggregated(tmp_path):
    """
    Verify that repeated paper-to-paper citations aggregate into a single weighted author edge.
    
    Arrange: Two identical citation lines P1 -> P2 and one author per paper.
    Act: Build networks.
    Assert: The directed edge (A -> B) has weight 2.
    """
    edges_file = tmp_path / "edges.txt"
    edges_file.write_text("P1 P2\nP1 P2\n", encoding="utf-8")

    paper_to_authors = {
        "P1": ["Author A"],
        "P2": ["Author B"],
    }

    _, G_cit = build_networks(str(edges_file), paper_to_authors)

    assert G_cit.has_edge("Author A", "Author B")
    assert G_cit["Author A"]["Author B"]["weight"] == 2


def test_build_networks_missing_edges_file(tmp_path):
    """
    Verify graceful handling when the citation edges file is missing.
    
    Arrange: A non-existent edge file path and a small author dictionary.
    Act: Build networks.
    Assert: Co-authorship graph is still constructed; citation graph is empty.
    """
    missing_file = tmp_path / "does_not_exist.txt"
    paper_to_authors = {"P1": ["Alice"], "P2": ["Bob"]}

    G_co, G_cit = build_networks(str(missing_file), paper_to_authors)

    assert G_co.number_of_nodes() == 2
    assert G_cit.number_of_nodes() == 0
    assert G_cit.number_of_edges() == 0
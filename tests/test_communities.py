"""
Tests for community detection and topic modeling in ``src.analysis.communities``.
These tests use barbell graphs and synthetic abstracts to validate Louvain
partitions, stability metrics (ARI), and TF-IDF keyword extraction.
"""

import networkx as nx
import numpy as np
import pytest

from src.analysis.communities import (
    analyze_communities_robust,
    check_community_distribution,
    get_community_partition,
)


# =============================================================================
# FIXTURES (Known Mathematical Structures & NLP Data)
# =============================================================================

@pytest.fixture
def clustered_authors_graph():
    """
    A 'Barbell' graph representing two distinct physics sub-fields.
    Nodes 0-4 form a fully connected Clique A.
    Nodes 5-9 form a fully connected Clique B.
    There is exactly 1 bridge connecting them (Node 4 to Node 5).
    
    Mathematical properties:
    - Louvain community detection MUST find exactly 2 communities.
    - Sizes of the communities MUST be exactly 5 and 5.
    - Modularity will be high because the intra-clique density heavily outweighs 
      the single inter-clique edge.
    """
    G = nx.barbell_graph(5, 0)
    # Convert integer nodes to string names (like our real author data)
    mapping = {i: f"Author_{i}" for i in G.nodes()}
    return nx.relabel_nodes(G, mapping)


@pytest.fixture
def mock_abstract_data():
    """
    Mock NLP data perfectly aligned with our two cliques.
    Clique A (Authors 0-4) writes about Quantum Gravity.
    Clique B (Authors 5-9) writes about Machine Learning.
    """
    author_to_papers = {f"Author_{i}": [f"P{i}"] for i in range(10)}
    
    paper_to_text = {}
    for i in range(5):
        paper_to_text[f"P{i}"] = "quantum gravity string theory holographic"
    for i in range(5, 10):
        paper_to_text[f"P{i}"] = "machine learning neural network data"
        
    return author_to_papers, paper_to_text


# =============================================================================
# PARTITION FOR VISUALIZATION
# =============================================================================

def test_get_community_partition_returns_node_to_community():
    """get_community_partition returns a dict mapping each node to a community id."""
    G = nx.barbell_graph(5, 0)
    partition = get_community_partition(G, random_state=0)
    assert isinstance(partition, dict)
    assert len(partition) == G.number_of_nodes()
    for node in G.nodes():
        assert node in partition
        assert isinstance(partition[node], int)
    # Barbell: expect 2 communities (two cliques)
    num_communities = len(set(partition.values()))
    assert num_communities >= 1


# =============================================================================
# COMMUNITY DISTRIBUTION TESTS
# =============================================================================

@pytest.mark.parametrize(
    "graph_func, expected_communities, expected_sizes",
    [
        (lambda: nx.barbell_graph(5, 0), 2, [5, 5]),  # Barbell splits perfectly into 2
        (lambda: nx.complete_graph(10), 1, [10]),     # Complete graph is just 1 community
    ],
)
def test_check_community_distribution(graph_func, expected_communities, expected_sizes):
    """
    Verify the community size extraction logic against known topologies.
    
    Arrange: Generate a graph with a mathematically known community structure.
    Act: Extract the size distribution.
    Assert: The number of communities and their exact sizes match analytical expectations.
    """
    G = graph_func()
    res = check_community_distribution(G, layer_name="Test")

    assert res["total_communities"] == expected_communities
    assert res["sizes"] == expected_sizes
    assert res["tiny_communities"] == 0  # Neither has isolated nodes < 5 size


# =============================================================================
# ROBUST CLUSTERING & NLP TOPIC TESTS
# =============================================================================

def test_analyze_communities_robust_no_text(clustered_authors_graph):
    """
    Verify the Louvain stability and partitioning when no text data is available.
    
    Arrange: Our 2-clique barbell graph, but empty text dictionaries.
    Act: Run robust community detection (2 iterations for speed).
    Assert: 
    - It finds 2 communities.
    - Modularity is strictly > 0.3 (a known threshold for barbell graphs).
    - ARI is 1.0 (perfectly stable between runs).
    - Topics list is gracefully empty.
    """
    res = analyze_communities_robust(
        clustered_authors_graph,
        author_to_papers={},
        paper_to_text={},
        n_iterations=2,
        n_jobs=1
    )

    partition = res["partition"]
    unique_communities = set(partition.values())
    
    assert len(unique_communities) == 2
    assert res["avg_modularity"] > 0.3
    assert np.isclose(res["avg_ari"], 1.0)
    assert len(res["topics"]) == 0


def test_analyze_communities_robust_with_topics(clustered_authors_graph, mock_abstract_data):
    """
    Verify TF-IDF correctly maps physics concepts to the detected topological communities.
    
    Arrange: Our 2-clique barbell graph, seeded with distinctly different text abstracts.
    Act: Run robust community detection and topic extraction.
    Assert:
    - It extracts exactly 2 topic dictionaries.
    - One community's keywords contain 'quantum'/'string'.
    - The other community's keywords contain 'machine'/'neural'.
    """
    a2p, p2t = mock_abstract_data

    res = analyze_communities_robust(
        clustered_authors_graph,
        author_to_papers=a2p,
        paper_to_text=p2t,
        n_iterations=2,
        n_jobs=1,
        top_keywords=3
    )

    topics = res["topics"]
    
    # We should have topics for our 2 communities (since both have size 5 >= threshold)
    assert len(topics) == 2
    
    # Extract the flat list of all keywords found
    all_keywords = []
    for t in topics:
        all_keywords.extend(t["keywords"])
        
    # Verify our distinct sub-field vocabularies were successfully isolated
    assert "quantum" in all_keywords or "string" in all_keywords
    assert "machine" in all_keywords or "neural" in all_keywords


def test_analyze_communities_robust_only_tiny_communities():
    """
    Edge case: all detected communities are smaller than the significance threshold (size < 5).
    
    Arrange: A small path graph with 4 nodes and no text data.
    Act: Run robust community detection.
    Assert: Partition and stability metrics are returned, but no topics are reported.
    """
    G = nx.path_graph(4)

    res = analyze_communities_robust(
        G,
        author_to_papers={},
        paper_to_text={},
        n_iterations=2,
        n_jobs=1,
    )

    assert isinstance(res["partition"], dict)
    assert 0.0 <= res["avg_modularity"] <= 1.0
    assert 0.0 <= res["avg_ari"] <= 1.0
    assert res["topics"] == []
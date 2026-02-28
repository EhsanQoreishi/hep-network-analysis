"""
Smoke tests for the top-level CLI orchestrator in ``main.py``. These tests
monkeypatch preprocessing, network construction, and visualization so that the
end-to-end pipeline can be exercised quickly on tiny in-memory graphs while
still producing the expected output artifacts.
"""

import os

import networkx as nx
import pytest

import main as cli


def test_main_smoke(monkeypatch, tmp_path):
    """
    Smoke-test the CLI orchestrator with minimal, fully controlled data.
    
    Arrange: Patch preprocessing, network construction, and visualization to operate on
    tiny in-memory graphs and temporary paths. Create dummy data/abstract paths so that
    path checks pass.
    Act: Invoke main() with custom CLI arguments.
    Assert: The run completes without error and produces an interactive map file.
    """

    # Prepare dummy data/abstracts paths expected by the CLI
    data_file = tmp_path / "edges.txt"
    data_file.write_text("P1 P2\n", encoding="utf-8")
    abstracts_dir = tmp_path / "abstracts"
    abstracts_dir.mkdir()

    output_dir = tmp_path / "results"

    # Patch core building blocks to avoid heavy real-data processing
    def fake_parse_abstracts(root_dir, n_jobs=-1):
        paper_to_authors = {"P1": ["Author A"], "P2": ["Author B"]}
        paper_to_text = {"P1": "dummy abstract text", "P2": "more text"}
        author_to_papers = {"Author A": ["P1"], "Author B": ["P2"]}
        return paper_to_authors, paper_to_text, author_to_papers

    def fake_build_networks(edges_file, paper_to_authors):
        G_co = nx.Graph()
        G_co.add_edge("Author A", "Author B", weight=1.0, distance=1.0)
        G_cit = nx.DiGraph()
        G_cit.add_edge("Author A", "Author B", weight=1.0)
        return G_co, G_cit

    def fake_visualize_network(G, title, partition=None, degree_full=None):
        os.makedirs(os.path.dirname(title), exist_ok=True)
        with open(title, "w", encoding="utf-8") as f:
            f.write("<html></html>")

    monkeypatch.setattr(cli, "parse_abstracts", fake_parse_abstracts)
    monkeypatch.setattr(cli, "build_networks", fake_build_networks)
    monkeypatch.setattr(cli, "visualize_network", fake_visualize_network)

    # Wire in fake CLI arguments
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "hep-network-analysis",
            "--data",
            str(data_file),
            "--abstracts",
            str(abstracts_dir),
            "--output",
            str(output_dir),
            "--jobs",
            "1",
        ],
    )

    # Act: run the orchestrator (will raise SystemExit only on hard failure)
    cli.main()

    # Assert: interactive HTML map (orchestrator final output) was produced
    html_map = output_dir / "interactive_map.html"
    assert html_map.exists()


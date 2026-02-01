import logging
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


def build_networks(
    edges_file: str, paper_to_authors: Dict[str, List[str]]
) -> Tuple[nx.Graph, nx.DiGraph]:
    """
    Constructs Co-authorship (Layer 1) and Citation (Layer 2) networks.

    Optimized: Uses Pandas relational merges for vector-speed citation graph construction,
    replacing slow iterative loops.

    Args:
        edges_file (str): Path to the citation edges file.
        paper_to_authors (Dict): Mapping of paper ID to list of author names.

    Returns:
        Tuple[nx.Graph, nx.DiGraph]: The co-authorship graph and the citation digraph.
    """
    logger.info("Building Author Networks...")

    all_authors = sorted(
        list({a for auths in paper_to_authors.values() for a in auths})
    )

    author_to_idx = {name: i for i, name in enumerate(all_authors)}
    paper_ids = sorted(paper_to_authors.keys())
    paper_to_idx = {pid: i for i, pid in enumerate(paper_ids)}

    rows = []
    cols = []
    data = []

    for pid, authors in paper_to_authors.items():
        if len(authors) < 2:
            continue
        p_idx = paper_to_idx[pid]
        for auth in authors:
            if auth in author_to_idx:
                rows.append(author_to_idx[auth])
                cols.append(p_idx)
                data.append(1)

    B = csr_matrix((data, (rows, cols)), shape=(len(all_authors), len(paper_ids)))

    C = B.dot(B.T)
    C.setdiag(0)
    C.eliminate_zeros()

    logger.info(f"Co-authorship matrix shape: {C.shape} with {C.nnz} edges")

    G_co = nx.from_scipy_sparse_array(C)
    nx.relabel_nodes(G_co, {i: name for i, name in enumerate(all_authors)}, copy=False)

    for u, v, d in G_co.edges(data=True):
        w = d.get("weight", 1)
        d["distance"] = 1.0 / w if w > 0 else 1.0

    logger.info("Processing citation edges (Vectorized)...")

    try:
        df_edges = pd.read_csv(
            edges_file, sep=r"\s+", comment="#", names=["source", "target"], dtype=str
        )

        valid_papers = set(paper_to_authors.keys())
        df_edges = df_edges[
            df_edges["source"].isin(valid_papers)
            & df_edges["target"].isin(valid_papers)
        ]

        flat_paper_authors = [
            (pid, auth) for pid, authors in paper_to_authors.items() for auth in authors
        ]

        df_authors = pd.DataFrame(flat_paper_authors, columns=["paper_id", "author"])

        merged_source = df_edges.merge(
            df_authors, left_on="source", right_on="paper_id"
        ).rename(columns={"author": "source_author"})

        merged_full = merged_source.merge(
            df_authors, left_on="target", right_on="paper_id"
        ).rename(columns={"author": "target_author"})

        merged_full = merged_full[
            merged_full["source_author"] != merged_full["target_author"]
        ]

        citation_weights = (
            merged_full.groupby(["source_author", "target_author"])
            .size()
            .reset_index(name="weight")
        )

        logger.info(
            f"Constructing Citation Graph from {len(citation_weights)} unique weighted edges..."
        )

        G_cit = nx.from_pandas_edgelist(
            citation_weights,
            source="source_author",
            target="target_author",
            edge_attr="weight",
            create_using=nx.DiGraph,
        )

    except FileNotFoundError:
        logger.error(f"Could not find {edges_file}. Returning empty citation graph.")
        return G_co, nx.DiGraph()
    except Exception as e:
        logger.error(f"Error processing citation file: {e}")
        return G_co, nx.DiGraph()

    return G_co, G_cit

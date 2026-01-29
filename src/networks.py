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

    Args:
        edges_file (str): Path to the citation edges file.
        paper_to_authors (Dict): Mapping of paper ID to list of author names.

    Returns:
        Tuple[nx.Graph, nx.DiGraph]: The co-authorship graph and the citation digraph.
    """
    logger.info("Building Author Networks...")

    # --- Layer 1: Co-authorship ---
    # Optimized: Flatten the author lists once to build the sparse matrix
    all_authors = sorted(
        list({a for auths in paper_to_authors.values() for a in auths})
    )

    author_to_idx = {name: i for i, name in enumerate(all_authors)}
    paper_ids = sorted(paper_to_authors.keys())
    paper_to_idx = {pid: i for i, pid in enumerate(paper_ids)}

    rows = []
    cols = []
    data = []

    # Flatten the (paper -> authors) structure into coordinate lists
    for pid, authors in paper_to_authors.items():
        if len(authors) < 2:
            continue
        p_idx = paper_to_idx[pid]
        for auth in authors:
            if auth in author_to_idx:
                rows.append(author_to_idx[auth])
                cols.append(p_idx)
                data.append(1)

    # Create Biadjacency Matrix B (Authors x Papers)
    B = csr_matrix((data, (rows, cols)), shape=(len(all_authors), len(paper_ids)))

    # Project to Author-Author network: C = B * B.T
    # This creates a clique for every paper
    C = B.dot(B.T)
    C.setdiag(0)
    C.eliminate_zeros()

    logger.info(f"Co-authorship matrix shape: {C.shape} with {C.nnz} edges")

    G_co = nx.from_scipy_sparse_array(C)
    nx.relabel_nodes(G_co, {i: name for i, name in enumerate(all_authors)}, copy=False)

    # Vectorized attribute setting isn't directly supported by nx,
    # but we can do it efficiently
    for u, v, d in G_co.edges(data=True):
        w = d.get("weight", 1)
        d["distance"] = 1.0 / w if w > 0 else 1.0

    # --- Layer 2: Citation ---
    logger.info("Processing citation edges...")
    G_cit = nx.DiGraph()

    try:
        # Vectorized Read: Use pandas to read the edge list (much faster than loop)
        # Assuming space-separated or tab-separated file with no header
        # Skipping lines starting with '#'
        df_edges = pd.read_csv(
            edges_file, sep=r"\s+", comment="#", names=["source", "target"], dtype=str
        )

        # Filter edges where both papers exist in our abstract dataset
        # (This avoids key errors later)
        valid_papers = set(paper_to_authors.keys())
        df_edges = df_edges[
            df_edges["source"].isin(valid_papers)
            & df_edges["target"].isin(valid_papers)
        ]

        logger.info(f"Found {len(df_edges)} valid citation links between papers.")

        # Build Author-Citation edges
        # We iterate over the valid paper-paper links
        for _, row in df_edges.iterrows():
            source_pid = row["source"]
            target_pid = row["target"]

            s_auths = paper_to_authors[source_pid]
            t_auths = paper_to_authors[target_pid]

            # Add edges for all author pairs (s -> t)
            # This logic assumes if Paper A cites Paper B, ALL authors of A cite ALL authors of B
            for sa in s_auths:
                for ta in t_auths:
                    if sa == ta:
                        continue

                    if G_cit.has_edge(sa, ta):
                        G_cit[sa][ta]["weight"] += 1.0
                    else:
                        G_cit.add_edge(sa, ta, weight=1.0)

    except FileNotFoundError:
        logger.error(f"Could not find {edges_file}. Returning empty citation graph.")
        return G_co, nx.DiGraph()
    except Exception as e:
        logger.error(f"Error processing citation file: {e}")
        return G_co, nx.DiGraph()

    return G_co, G_cit

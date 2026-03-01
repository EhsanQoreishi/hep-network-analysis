import logging
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

def build_networks(edges_file: str, paper_to_authors: Dict[str, List[str]]) -> Tuple[nx.Graph, nx.DiGraph]:
    """
    Build co-authorship (B*B^T sparse projection) and citation (vectorized pandas) networks.
    Returns (G_co, G_cit). Edges get 'weight' and 'distance' (1/weight) for path metrics.
    """
    logger.info("Building Author Networks...")

    # =========================================================================
    # Part 1: Bipartite Projection for Co-authorship Network
    # =========================================================================
    all_authors = sorted({a for auths in paper_to_authors.values() for a in auths})
    author_to_idx = {name: i for i, name in enumerate(all_authors)}
    paper_ids = sorted(paper_to_authors.keys())
    paper_to_idx = {pid: i for i, pid in enumerate(paper_ids)}

    rows: List[int] = []
    cols: List[int] = []
    data: List[int] = []

    # Build the Author-Paper incidence matrix
    for pid, authors in paper_to_authors.items():
        if len(authors) < 2:
            continue
        p_idx = paper_to_idx[pid]
        for auth in authors:
            i = author_to_idx.get(auth)
            if i is not None:
                rows.append(i)
                cols.append(p_idx)
                data.append(1)

    # B is an (Authors x Papers) matrix. B * B^T yields the (Authors x Authors) adjacency matrix.
    B = csr_matrix((data, (rows, cols)), shape=(len(all_authors), len(paper_ids)))
    C = B.dot(B.T)
    C.setdiag(0)  # Remove self-loops (an author cannot co-author with themselves)
    C.eliminate_zeros()

    logger.info(f"Co-authorship matrix shape: {C.shape} with {C.nnz} edges")

    G_co = nx.from_scipy_sparse_array(C)
    nx.relabel_nodes(G_co, {i: name for i, name in enumerate(all_authors)}, copy=False)

    # Invert weights to create a 'distance' metric (more collaborations = closer distance)
    # This is crucial for correctly computing betweenness centrality later.
    for _, _, d in G_co.edges(data=True):
        w = d.get("weight", 1)
        try:
            wf = float(w)
            d["distance"] = (1.0 / wf) if wf > 0 else 1.0
        except Exception:
            d["distance"] = 1.0

    # =========================================================================
    # Part 2: Vectorized Edge Mapping for Citation Network
    # =========================================================================
    logger.info("Processing citation edges (Vectorized)...")

    try:
        df_edges = pd.read_csv(edges_file, sep=r"\s+", comment="#", names=["source", "target"], dtype=str)

        # Filter out citations to/from papers that don't exist in our parsed abstracts
        valid_papers = set(paper_to_authors.keys())
        df_edges = df_edges[df_edges["source"].isin(valid_papers) & df_edges["target"].isin(valid_papers)]

        # Flatten the dictionary into a dataframe for joining
        df_authors = pd.DataFrame(
            [(pid, auth) for pid, authors in paper_to_authors.items() for auth in authors],
            columns=["paper_id", "author"],
        )

        # Map Paper A -> Paper B into Author(s) of A -> Author(s) of B
        merged_source = df_edges.merge(df_authors, left_on="source", right_on="paper_id").rename(
            columns={"author": "source_author"}
        )

        merged_full = merged_source.merge(df_authors, left_on="target", right_on="paper_id").rename(
            columns={"author": "target_author"}
        )

        # Filter out self-citations (where an author cites their own past work)
        # We only want to measure external intellectual influence.
        merged_full = merged_full[merged_full["source_author"] != merged_full["target_author"]]

        # Aggregate multiple citations between the same pair of authors into a single weighted edge
        citation_weights = merged_full.groupby(["source_author", "target_author"]).size().reset_index(name="weight")

        logger.info(f"Constructing Citation Graph from {len(citation_weights)} unique weighted edges...")

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
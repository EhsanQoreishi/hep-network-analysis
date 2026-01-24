import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
from typing import Dict, List, Tuple

def build_networks(edges_file: str, paper_to_authors: Dict[str, List[str]]) -> Tuple[nx.Graph, nx.DiGraph]:
    """Constructs Co-authorship (Layer 1) and Citation (Layer 2) networks."""
    print("Building Author Networks...")

    # --- Layer 1: Co-authorship ---
    all_authors = set()
    for auths in paper_to_authors.values():
        all_authors.update(auths)
    all_authors = sorted(list(all_authors))
    
    author_to_idx = {name: i for i, name in enumerate(all_authors)}
    idx_to_author = {i: name for i, name in enumerate(all_authors)}
    paper_to_idx = {pid: i for i, pid in enumerate(sorted(paper_to_authors.keys()))}

    rows, cols, data = [], [], []
    for pid, authors in paper_to_authors.items():
        if len(authors) < 2: continue
        p_idx = paper_to_idx[pid]
        for auth in authors:
            if auth in author_to_idx:
                rows.append(author_to_idx[auth])
                cols.append(p_idx)
                data.append(1)

    B = csr_matrix((data, (rows, cols)), shape=(len(all_authors), len(paper_to_idx)))
    C = B.dot(B.T)
    C.setdiag(0)
    C.eliminate_zeros()

    G_co = nx.from_scipy_sparse_array(C)
    nx.relabel_nodes(G_co, idx_to_author, copy=False)

    # Calculate effective distance (1/w)
    for u, v, d in G_co.edges(data=True):
        w = d.get('weight', 1)
        d['distance'] = 1.0 / w if w > 0 else 1.0

    # --- Layer 2: Citation ---
    print("Processing citation edges...")
    G_cit = nx.DiGraph()
    try:
        with open(edges_file, "r") as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.split()
                if len(parts) < 2: continue
                
                s_auths = paper_to_authors.get(parts[0], [])
                t_auths = paper_to_authors.get(parts[1], [])
                
                if s_auths and t_auths:
                    for sa in s_auths:
                        for ta in t_auths:
                            if sa == ta: continue
                            if G_cit.has_edge(sa, ta):
                                G_cit[sa][ta]["weight"] += 1.0
                            else:
                                G_cit.add_edge(sa, ta, weight=1.0)
    except FileNotFoundError:
        print(f"Error: Could not find {edges_file}")
        return None, None

    return G_co, G_cit
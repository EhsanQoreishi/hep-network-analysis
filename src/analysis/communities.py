import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List

import networkx as nx
import numpy as np
from community import community_louvain
from joblib import Parallel, delayed
from scipy.sparse import diags
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import adjusted_rand_score

from src.constants import CUSTOM_STOP_WORDS

logger = logging.getLogger(__name__)

def _run_louvain(G: nx.Graph, seed: int) -> Dict[str, int]:
    """
    Helper function to execute the Louvain community detection algorithm.
    Isolated here so we can easily run it in parallel across multiple CPU cores.
    """
    return community_louvain.best_partition(G, random_state=seed)


def get_community_partition(G: nx.Graph, random_state: int = 0) -> Dict[str, int]:
    """
    Returns a node-to-community-id mapping for the given graph using Louvain.
    Used by the visualization layer to color nodes; analysis stays in this module.
    """
    G_undir = G.to_undirected() if G.is_directed() else G
    return community_louvain.best_partition(G_undir, random_state=random_state)


def check_community_distribution(G: nx.Graph, layer_name: str = "Network") -> Dict[str, Any]:
    """
    Louvain community size distribution and summary stats (tiny count, top-5 sizes).
    """
    logger.info(f"Checking Community Size Distribution ({layer_name})...")

    G_undir = G.to_undirected()
    partition = community_louvain.best_partition(G_undir, random_state=0)
    size_counts = Counter(partition.values())
    sizes = sorted(list(size_counts.values()), reverse=True)

    logger.info(f"Total Communities: {len(sizes)}")
    logger.info(f"Top 5 Largest (Major Fields): {sizes[:5]}")

    tiny_communities = sum(1 for s in sizes if s < 5)
    logger.info(f"Number of 'Tiny' Communities (Size < 5): {tiny_communities}")

    return {
        "sizes": sizes,
        "total_communities": len(sizes),
        "top_5_sizes": sizes[:5],
        "tiny_communities": tiny_communities,
    }

def analyze_communities_robust(
    G: nx.Graph,
    author_to_papers: Dict[str, List[str]],
    paper_to_text: Dict[str, str],
    layer_name: str = "Network",
    n_iterations: int = 10,
    n_jobs: int = -1,
    top_k_report: int = 3,
    top_keywords: int = 8,
) -> Dict[str, Any]:
    """
    Louvain over n_iterations; reports ARI stability and TF-IDF keywords per community.
    """
    logger.info(f"Starting Robust Community Detection & Stability Analysis ({layer_name})...")

    G_undir = G.to_undirected()
    logger.info(f"Running Louvain {n_iterations} times (Parallel n_jobs={n_jobs})...")

    # Parallel execution to handle high computational load of multiple iterations
    partitions_list = Parallel(n_jobs=n_jobs)(delayed(_run_louvain)(G_undir, i) for i in range(n_iterations))

    modularities = [community_louvain.modularity(part, G_undir) for part in partitions_list]

    nodes = list(G_undir.nodes())

    # ARI over all (n_iterations choose 2) pairs of runs (report Eq. 6.3)
    ari_scores = []
    for r in range(n_iterations):
        for rp in range(r + 1, n_iterations):
            labels_r = [partitions_list[r][n] for n in nodes]
            labels_rp = [partitions_list[rp][n] for n in nodes]
            ari_scores.append(adjusted_rand_score(labels_r, labels_rp))
    avg_ari = float(np.mean(ari_scores)) if ari_scores else 1.0
    avg_modularity = float(np.mean(modularities)) if modularities else 0.0

    logger.info("Stability Results:")
    logger.info(f"  Average Modularity (Q): {avg_modularity:.4f}")
    logger.info(f"  Stability (Avg ARI):    {avg_ari:.4f}")
    logger.info(f"  Runs (n_iterations):    {int(n_iterations)}")

    # Select the partition that maximized modularity
    best_idx = int(np.argmax(modularities)) if modularities else 0
    best_partition = partitions_list[best_idx]

    communities = defaultdict(list)
    for node, comm_id in best_partition.items():
        communities[comm_id].append(node)

    # Filter for significant research groups (size >= 5) to avoid noise from tiny teams
    significant_comms = {cid: auths for cid, auths in communities.items() if len(auths) >= 5}
    sorted_comm_ids = sorted(significant_comms.keys(), key=lambda k: len(significant_comms[k]), reverse=True)

    community_documents: List[str] = []
    map_index_to_comm_id: List[int] = []

    # Aggregate abstract text for each community to create a "field profile"
    for comm_id in sorted_comm_ids:
        comm_text_list = [
            paper_to_text[pid]
            for author in significant_comms[comm_id]
            for pid in author_to_papers.get(author, [])
            if pid in paper_to_text
        ]
        full_text = " ".join(comm_text_list)
        if full_text.strip():
            community_documents.append(full_text)
            map_index_to_comm_id.append(comm_id)

    topic_results = []
    
    if not community_documents:
        logger.warning("No abstract text available for topic modeling. Returning partition only.")
    else:
        stop_words = list(ENGLISH_STOP_WORDS.union(CUSTOM_STOP_WORDS))

        # Report Eqs. 6.5–6.7: tf(t,d) = f(t,d)/sum_t' f(t',d), idf(t) = log((1+D)/(1+df(t)))+1
        count_vec = CountVectorizer(stop_words=stop_words, max_features=1000, max_df=1.0)
        X = count_vec.fit_transform(community_documents)
        feature_names = np.array(count_vec.get_feature_names_out())
        n_docs, n_terms = X.shape
        D = n_docs

        # tf: normalized term frequency per document
        row_sums = np.array(X.sum(axis=1)).flatten()
        row_sums = np.maximum(row_sums, 1)
        tf_matrix = diags(1.0 / row_sums) @ X
        # idf: log((1+D)/(1+df(t)))+1
        df = np.array((X > 0).sum(axis=0)).flatten()
        idf_vec = np.log((1.0 + D) / (1.0 + df)) + 1.0
        tfidf_matrix = tf_matrix @ diags(idf_vec)

        try:
            logger.info(f"--- Top Topics per Community ({layer_name}) ---")
            for i, comm_id in enumerate(map_index_to_comm_id[:top_k_report]):
                size = len(significant_comms[comm_id])
                row = tfidf_matrix[i]
                scores = row.toarray().flatten()
                top_indices = scores.argsort()[::-1][:top_keywords]
                top_keywords_list = feature_names[top_indices]
                
                keywords_str = ", ".join(top_keywords_list)
                logger.info(
                    f"Community {comm_id} (Size: {size}) - Top TF-IDF Keywords (Most Specific): {keywords_str}"
                )
                
                topic_results.append({
                    "layer": layer_name,
                    "community_id": comm_id,
                    "community_size": size,
                    "keywords": top_keywords_list.tolist()
                })
                
            logger.info(f"Reported communities: top_k_report={int(top_k_report)} (largest by size among communities with size>=5)")

        except ValueError as e:
            logger.error(f"Skipping topic modeling (not enough text data): {e}")

    return {
        "partition": best_partition,
        "avg_modularity": avg_modularity,
        "avg_ari": avg_ari,
        "topics": topic_results
    }
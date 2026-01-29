import logging
import os
from collections import Counter, defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from community import community_louvain
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score

from src.constants import CUSTOM_STOP_WORDS

logger = logging.getLogger(__name__)


def check_community_distribution(G: nx.Graph, output_dir: str = "results") -> None:
    """
    Identifies communities using the Louvain method and analyzes their size distribution.

    Args:
        G (nx.Graph): The network graph.
        output_dir (str): Directory to save the distribution plot.
    """
    logger.info("Checking Community Size Distribution...")

    partition = community_louvain.best_partition(G)
    size_counts = Counter(partition.values())
    sizes = sorted(list(size_counts.values()), reverse=True)

    logger.info(f"Total Communities: {len(sizes)}")
    logger.info(f"Top 5 Largest (Major Fields): {sizes[:5]}")

    tiny_communities = sum(1 for s in sizes if s < 5)
    logger.info(f"Number of 'Tiny' Communities (Size < 5): {tiny_communities}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(sizes, bins=50, color="teal", edgecolor="black")
    plt.title("Distribution of Community Sizes (Log Scale)")
    plt.xlabel("Number of Authors in Community")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.grid(axis="y", alpha=0.5)

    save_path = os.path.join(output_dir, "community_size_distribution.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Distribution plot saved to {save_path}")


def analyze_communities_robust(
    G: nx.Graph,
    author_to_papers: Dict[str, List[str]],
    paper_to_text: Dict[str, str],
    n_iterations: int = 10,
) -> Dict[str, int]:
    """
    Performs robust community detection (Louvain) with stability testing and Topic Modeling.

    Args:
        G (nx.Graph): The network graph.
        author_to_papers (Dict): Mapping of author names to paper IDs.
        paper_to_text (Dict): Mapping of paper IDs to abstract text.
        n_iterations (int): Number of runs to test Louvain stability.

    Returns:
        Dict[str, int]: The best partition map (node -> community ID).
    """
    logger.info("Starting Robust Community Detection & Stability Analysis...")

    G_undir = G.to_undirected()
    logger.info(f"Running Louvain {n_iterations} times to test stability...")

    # Run partitions and store modularities
    partitions_list = []
    modularities = []

    for i in range(n_iterations):
        # random_state ensures reproducibility for specific runs if needed
        part = community_louvain.best_partition(G_undir, random_state=i)
        partitions_list.append(part)
        q = community_louvain.modularity(part, G_undir)
        modularities.append(q)

    # Stability Check (ARI Score)
    nodes = list(G_undir.nodes())
    first_run_labels = [partitions_list[0][n] for n in nodes]

    ari_scores = [
        adjusted_rand_score(first_run_labels, [partitions_list[i][n] for n in nodes])
        for i in range(1, n_iterations)
    ]

    avg_ari = np.mean(ari_scores) if ari_scores else 1.0
    avg_modularity = np.mean(modularities)

    logger.info("Stability Results:")
    logger.info(f"  Average Modularity (Q): {avg_modularity:.4f}")
    logger.info(f"  Stability (Avg ARI):    {avg_ari:.4f}")

    if avg_modularity > 0.4:
        logger.info("  -> Strong community structure found (Q > 0.4).")
    else:
        logger.warning(
            "  -> Weak community structure (Q < 0.4). Network may be well-mixed."
        )

    # Select best partition
    best_idx = np.argmax(modularities)
    best_partition = partitions_list[best_idx]

    # Invert partition for topic modeling: CommID -> [Authors]
    communities = defaultdict(list)
    for node, comm_id in best_partition.items():
        communities[comm_id].append(node)

    # Filter for significant communities (>= 5 members)
    significant_comms = {
        cid: auths for cid, auths in communities.items() if len(auths) >= 5
    }

    sorted_comm_ids = sorted(
        significant_comms.keys(), key=lambda k: len(significant_comms[k]), reverse=True
    )

    # Aggregate text per community
    community_documents = []
    map_index_to_comm_id = []

    for comm_id in sorted_comm_ids:
        # Optimized list comprehension for text aggregation
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

    if not community_documents:
        logger.warning(
            "No abstract text available for topic modeling. Returning partition only."
        )
        return best_partition

    # TF-IDF Topic Modeling
    stop_words = list(ENGLISH_STOP_WORDS.union(CUSTOM_STOP_WORDS))
    tfidf = TfidfVectorizer(
        stop_words=stop_words, max_features=1000, max_df=0.25, sublinear_tf=True
    )

    try:
        tfidf_matrix = tfidf.fit_transform(community_documents)
        feature_names = np.array(tfidf.get_feature_names_out())

        logger.info("--- Top Topics per Community ---")
        for i, comm_id in enumerate(map_index_to_comm_id[:5]):
            size = len(significant_comms[comm_id])
            row = tfidf_matrix[i]
            scores = row.toarray().flatten()

            # Vectorized sort to find top keywords
            top_indices = scores.argsort()[::-1][:8]
            top_keywords = feature_names[top_indices]

            logger.info(
                f"Community {comm_id} (Size: {size}): {', '.join(top_keywords)}"
            )

    except ValueError as e:
        logger.error(f"Skipping topic modeling (not enough text data): {e}")

    return best_partition

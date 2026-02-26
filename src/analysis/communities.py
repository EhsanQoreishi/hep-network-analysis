import logging
import os
from collections import Counter, defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from community import community_louvain
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score

from src.constants import CUSTOM_STOP_WORDS

logger = logging.getLogger(__name__)

def _sanitize_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(":", "")
        .replace(",", "")
        .replace("__", "_")
    )

def _savefig(path: str) -> None:
    base, ext = os.path.splitext(path)
    if ext.lower() != ".pdf":
        path = base + ".pdf"
    plt.savefig(path, format="pdf", bbox_inches="tight")
    logger.info(f"Saved figure to: {path}")

def _run_louvain(G: nx.Graph, seed: int) -> Dict[str, int]:
    return community_louvain.best_partition(G, random_state=seed)

def check_community_distribution(G: nx.Graph, layer_name: str = "Network", output_dir: str = "results") -> None:
    logger.info(f"Checking Community Size Distribution ({layer_name})...")

    G_undir = G.to_undirected()
    partition = community_louvain.best_partition(G_undir, random_state=0)
    size_counts = Counter(partition.values())
    sizes = sorted(list(size_counts.values()), reverse=True)

    logger.info(f"Total Communities: {len(sizes)}")
    logger.info(f"Top 5 Largest (Major Fields): {sizes[:5]}")

    tiny_communities = sum(1 for s in sizes if s < 5)
    logger.info(f"Number of 'Tiny' Communities (Size < 5): {tiny_communities}")

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(sizes, bins=50, color="teal", edgecolor="black")
    plt.title(f"Community Size Distribution ({layer_name})")
    plt.xlabel("Community size (number of authors)")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.grid(axis="y", alpha=0.5)

    safe_layer = _sanitize_name(layer_name)
    save_path = os.path.join(output_dir, f"{safe_layer}_community_size_distribution.pdf")
    _savefig(save_path)
    plt.close()

def analyze_communities_robust(
    G: nx.Graph,
    author_to_papers: Dict[str, List[str]],
    paper_to_text: Dict[str, str],
    layer_name: str = "Network",
    n_iterations: int = 10,
    n_jobs: int = -1,
    output_dir: str = "results",
    top_k_report: int = 3,
    top_keywords: int = 8,
) -> Dict[str, int]:
    logger.info(f"Starting Robust Community Detection & Stability Analysis ({layer_name})...")

    G_undir = G.to_undirected()
    logger.info(f"Running Louvain {n_iterations} times (Parallel n_jobs={n_jobs})...")

    partitions_list = Parallel(n_jobs=n_jobs)(delayed(_run_louvain)(G_undir, i) for i in range(n_iterations))

    modularities = [community_louvain.modularity(part, G_undir) for part in partitions_list]

    nodes = list(G_undir.nodes())
    first_run_labels = [partitions_list[0][n] for n in nodes]

    ari_scores = [
        adjusted_rand_score(first_run_labels, [partitions_list[i][n] for n in nodes])
        for i in range(1, n_iterations)
    ]

    avg_ari = float(np.mean(ari_scores)) if ari_scores else 1.0
    avg_modularity = float(np.mean(modularities)) if modularities else 0.0

    logger.info("Stability Results:")
    logger.info(f"  Average Modularity (Q): {avg_modularity:.4f}")
    logger.info(f"  Stability (Avg ARI):    {avg_ari:.4f}")
    logger.info(f"  Runs (n_iterations):    {int(n_iterations)}")

    best_idx = int(np.argmax(modularities)) if modularities else 0
    best_partition = partitions_list[best_idx]

    communities = defaultdict(list)
    for node, comm_id in best_partition.items():
        communities[comm_id].append(node)

    significant_comms = {cid: auths for cid, auths in communities.items() if len(auths) >= 5}
    sorted_comm_ids = sorted(significant_comms.keys(), key=lambda k: len(significant_comms[k]), reverse=True)

    community_documents: List[str] = []
    map_index_to_comm_id: List[int] = []

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

    if not community_documents:
        logger.warning("No abstract text available for topic modeling. Returning partition only.")
        return best_partition

    stop_words = list(ENGLISH_STOP_WORDS.union(CUSTOM_STOP_WORDS))
    tfidf = TfidfVectorizer(stop_words=stop_words, max_features=1000, max_df=0.25, sublinear_tf=True)

    try:
        tfidf_matrix = tfidf.fit_transform(community_documents)
        feature_names = np.array(tfidf.get_feature_names_out())

        logger.info(f"--- Top Topics per Community ({layer_name}) ---")
        for i, comm_id in enumerate(map_index_to_comm_id[:top_k_report]):
            size = len(significant_comms[comm_id])
            row = tfidf_matrix[i]
            scores = row.toarray().flatten()
            top_indices = scores.argsort()[::-1][:top_keywords]
            top_keywords_list = feature_names[top_indices]
            logger.info(
                f"Community {comm_id} (Size: {size}) - Top TF-IDF Keywords (Most Specific): {', '.join(top_keywords_list)}"
            )

        os.makedirs(output_dir, exist_ok=True)
        safe_layer = _sanitize_name(layer_name)
        csv_path = os.path.join(output_dir, f"{safe_layer}_top_communities_tfidf.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("layer,community_id,community_size,keywords\n")
            for i, comm_id in enumerate(map_index_to_comm_id[:top_k_report]):
                size = len(significant_comms[comm_id])
                scores = tfidf_matrix[i].toarray().flatten()
                top_indices = scores.argsort()[::-1][:top_keywords]
                top_keywords_list = feature_names[top_indices]
                f.write(f"{layer_name},{comm_id},{size},\"{', '.join(top_keywords_list)}\"\n")
        logger.info(f"Saved topics table to: {csv_path}")
        logger.info(f"Reported communities: top_k_report={int(top_k_report)} (largest by size among communities with size>=5)")

    except ValueError as e:
        logger.error(f"Skipping topic modeling (not enough text data): {e}")

    return best_partition

import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from typing import Dict, List, Any

# External libraries
from community import community_louvain
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import adjusted_rand_score

def check_community_distribution(G: nx.Graph, output_dir: str = "results") -> None:
    """
    Identifies communities using the Louvain method and analyzes their size distribution.
    Checks if the network is fragmented or dominated by a giant component.
    """
    print("\n--- Community Size Distribution Check ---")

    # Generate partition based on modularity optimization
    partition = community_louvain.best_partition(G)

    size_counts = Counter(partition.values())
    sizes = sorted(list(size_counts.values()), reverse=True)

    print(f"Total Communities: {len(sizes)}")
    print(f"Top 5 Largest (Major Fields): {sizes[:5]}")
    print(f"Number of 'Tiny' Communities (Size < 5): {len([s for s in sizes if s < 5])}")

    # Plotting
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

def analyze_communities_robust(
    G: nx.Graph, 
    author_to_papers: Dict[str, List[str]], 
    paper_to_text: Dict[str, str], 
    n_iterations: int = 10
) -> Dict[str, int]:
    """
    Performs robust community detection (Louvain) with stability testing.
    Includes TF-IDF Topic Modeling to label the discovered groups.
    """
    print("\n--- Robust Community Detection & Stability Analysis ---")

    G_undir = G.to_undirected()
    print(f"Running Louvain {n_iterations} times to test stability...")

    partitions_list = []
    modularities = []

    # 1. Stability Loop
    for i in range(n_iterations):
        part = community_louvain.best_partition(G_undir, random_state=i)
        partitions_list.append(part)
        q = community_louvain.modularity(part, G_undir)
        modularities.append(q)

    # 2. ARI Calculation (Consistency Check)
    nodes = list(G_undir.nodes())
    first_run_labels = [partitions_list[0][n] for n in nodes]
    ari_scores = []

    for i in range(1, n_iterations):
        current_run_labels = [partitions_list[i][n] for n in nodes]
        score = adjusted_rand_score(first_run_labels, current_run_labels)
        ari_scores.append(score)

    avg_ari = np.mean(ari_scores) if ari_scores else 1.0
    avg_modularity = np.mean(modularities)

    print(f"\nStability Results:")
    print(f"  Average Modularity (Q): {avg_modularity:.4f}")
    print(f"  Stability (Avg ARI):    {avg_ari:.4f}")

    if avg_modularity > 0.4:
        print("  -> Strong community structure found (Q > 0.4).")
    else:
        print("  -> Weak community structure (Q < 0.4). Well-mixed network.")

    # Select best partition
    best_idx = np.argmax(modularities)
    best_partition = partitions_list[best_idx]

    # 3. Topic Modeling (TF-IDF)
    # Group authors by community
    communities = defaultdict(list)
    for node, comm_id in best_partition.items():
        communities[comm_id].append(node)

    # Filter for significant communities
    significant_comms = {cid: auths for cid, auths in communities.items() if len(auths) >= 5}
    sorted_comm_ids = sorted(significant_comms.keys(), key=lambda k: len(significant_comms[k]), reverse=True)

    # Aggregate text
    community_documents = []
    map_index_to_comm_id = []

    for comm_id in sorted_comm_ids:
        comm_text_parts = []
        for author in significant_comms[comm_id]:
            papers = author_to_papers.get(author, [])
            for pid in papers:
                if pid in paper_to_text:
                    comm_text_parts.append(paper_to_text[pid])
        
        full_text = " ".join(comm_text_parts)
        if full_text.strip():
            community_documents.append(full_text)
            map_index_to_comm_id.append(comm_id)

    if not community_documents:
        print("Warning: No abstract text available for topic modeling.")
        return best_partition

    # TF-IDF Vectorization
    custom_stop_words = [
        "theory", "model", "field", "physics", "results", "paper", "study", 
        "analysis", "high", "energy", "using", "based", "shown"
    ]
    stop_words = list(ENGLISH_STOP_WORDS.union(custom_stop_words))
    
    tfidf = TfidfVectorizer(
        stop_words=stop_words, max_features=1000, max_df=0.25, sublinear_tf=True
    )

    try:
        tfidf_matrix = tfidf.fit_transform(community_documents)
        feature_names = np.array(tfidf.get_feature_names_out())
        
        print("\n--- Top Topics per Community ---")
        # Show top 5 largest communities
        for i, comm_id in enumerate(map_index_to_comm_id[:5]):
            size = len(significant_comms[comm_id])
            row = tfidf_matrix[i]
            scores = row.toarray().flatten()
            top_indices = scores.argsort()[::-1][:8]
            top_keywords = feature_names[top_indices]
            
            print(f"Community {comm_id} (Size: {size}): {', '.join(top_keywords)}")
            
    except ValueError as e:
        print(f"Skipping topic modeling (not enough text data): {e}")

    return best_partition
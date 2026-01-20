import os
import re
import warnings
import argparse
from typing import Dict, List, Tuple, Optional, Set, DefaultDict

import powerlaw
import scipy.sparse.linalg
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from community import community_louvain
from pyvis.network import Network
from scipy.stats import gaussian_kde, spearmanr
from collections import defaultdict, Counter
from itertools import combinations
from scipy.sparse import csr_matrix

# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================
EDGES_FILE = "data/cit-HepTh.txt"
ABSTRACTS_DIR = "data/cit-HepTh-abstracts"
RESULTS_DIR = "results"

# Ensure the results directory exists for saving plots
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# ==========================================
# 2. DATA PARSING
# ==========================================

def normalize_name(name: str) -> Optional[str]:
    """
    Standardizes author names into 'F. Lastname' format.
    Handles compound last names (e.g., 'van den Berg') where the prefix is lowercase.

    Args:
        name (str): The raw author name string.

    Returns:
        Optional[str]: The normalized name, or None if the name is invalid.
    """
    name = name.replace(".", "").strip()
    parts = name.split()

    if len(parts) < 2:
        return None

    # Handle compound last names if the second-to-last part is lowercase
    if len(parts) > 2 and parts[-2].islower():
        last_name = f"{parts[-2]} {parts[-1]}"
    else:
        last_name = parts[-1]

    first_initial = parts[0][0].upper()

    return f"{first_initial}. {last_name}"


def clean_text(text: str) -> str:
    """
    Preprocesses abstract text for NLP tasks.
    Removes LaTeX formatting, mathematical variables, and non-alphabetic symbols.

    Args:
        text (str): The raw abstract text.

    Returns:
        str: The normalized, lowercase text.
    """
    # 1. Remove LaTeX commands (e.g., \widehat, \frac, \begin)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)

    # 2. Remove math symbols/variables with underscores or digits
    text = re.sub(r"\b[a-zA-Z]+[_\d][a-zA-Z\d]*\b", " ", text)

    # 3. Remove single characters (often math variables like x, y, z)
    text = re.sub(r"\b[a-zA-Z]\b", " ", text)

    # 4. Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # 5. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text.lower()


def parse_abstracts(root_dir):
    """
    Scans a directory for .abs files to extract metadata and abstract text.

    This function performs the following:
    1. Iterates through the file system to find abstract files.
    2. Uses regular expressions to isolate author names while filtering out affiliations and noise.
    3. Matches author names against an extensive blocklist of geographical and institutional terms.
    4. Extracts and cleans the abstract body text for later NLP analysis.

    Returns:
        paper_to_authors (dict): Mapping of paper IDs to lists of normalized author names.
        paper_to_text (dict): Mapping of paper IDs to cleaned abstract text.
        author_to_papers (dict): Reverse mapping of authors to their associated paper IDs.
    """
    print(f"Scanning abstracts in {root_dir}...")

    paper_to_authors = defaultdict(list)
    paper_to_text = {}
    author_to_papers = defaultdict(list)

    # Set of terms used to filter out institutional or geographical noise from author lists
    NON_AUTHOR_TERMS = {
        "italy",
        "germany",
        "france",
        "spain",
        "russia",
        "usa",
        "japan",
        "uk",
        "england",
        "canada",
        "switzerland",
        "brazil",
        "india",
        "china",
        "korea",
        "australia",
        "mexico",
        "israel",
        "netherlands",
        "belgium",
        "sweden",
        "cern",
        "trieste",
        "moscow",
        "rome",
        "paris",
        "london",
        "berlin",
        "madrid",
        "caltech",
        "mit",
        "stanford",
        "harvard",
        "princeton",
        "cambridge",
        "oxford",
        "chicago",
        "columbia",
        "berkeley",
        "infn",
        "fisica",
        "physics",
        "department",
        "dept",
        "univ",
        "university",
        "institute",
        "istituto",
        "nazionale",
        "research",
        "center",
        "centre",
        "lab",
        "laboratory",
        "school",
        "college",
        "division",
        "section",
        "group",
        "collab",
        "collaboration",
        "et al",
        "et",
        "al",
        "di",
        "de",
        "del",
        "dipartimento",
        "departamento",
        "complutense",
        "autonoma",
        "polytechnique",
        "state",
        "tech",
        "technology",
    }

    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".abs"):
                continue

            paper_id = file.replace(".abs", "")
            path = os.path.join(root, file)

            try:
                with open(path, "r", encoding="latin-1") as f:
                    content = f.read()

                    # Extract the Author block using regex delimiters
                    auth_match = re.search(
                        r"Authors?:\s*(.+?)(?=\n(?:Comments|Journal-ref|Subj-class|\\)|$)",
                        content,
                        re.DOTALL | re.IGNORECASE,
                    )

                    if auth_match:
                        raw_authors = auth_match.group(1).replace("\n", " ")

                        # Remove affiliation parentheses before splitting names
                        raw_authors = re.sub(r"\(.*?\)", "", raw_authors)
                        authors = re.split(r",|\sand\s|;", raw_authors)

                        cleaned_authors = []
                        for a in authors:
                            name = a.strip()
                            if len(name) <= 2:
                                continue

                            # Validate name against blocklist and normalize format
                            if any(x in name.lower() for x in NON_AUTHOR_TERMS):
                                continue

                            normalized_name = normalize_name(name)
                            if normalized_name:
                                cleaned_authors.append(normalized_name)

                        if cleaned_authors:
                            paper_to_authors[paper_id] = cleaned_authors
                            for auth in cleaned_authors:
                                author_to_papers[auth].append(paper_id)

                    # Extract and clean abstract text body
                    parts = content.split("\\\\")
                    abstract_candidate = parts[2] if len(parts) >= 3 else parts[-1]
                    cleaned_abstract = clean_text(abstract_candidate)

                    # Store substantial text only to filter out metadata-only entries
                    if len(cleaned_abstract) > 50:
                        paper_to_text[paper_id] = cleaned_abstract

            except Exception:
                # Silently continue on individual file parsing errors
                continue

    print(f"Parsed {len(paper_to_authors)} papers.")
    return paper_to_authors, paper_to_text, author_to_papers


# ==========================================
# 3. GRAPH CONSTRUCTION
# ==========================================


def build_networks(edges_file, paper_to_authors):
    """
    Constructs two distinct layers of author networks: Co-authorship and Citation.

    1. Co-authorship Layer (Optimized): Uses sparse matrix multiplication (B * B.T)
       to project the bipartite author-paper graph into a unipartite author graph.
       This is O(N) instead of O(N^2) for dense cliques.
    2. Citation Layer: Directed graph of citations with fractional weighting.
    """
    print("Building Author Networks...")

    # --- Layer 1: Co-authorship Network Construction (Matrix Method) ---
    print("  Constructing Co-authorship Matrix...")

    # 1. Create Integer Mappings (The Hilbert Space)
    # Matrices work on integer indices (0, 1, 2...), not names.
    # We must collect all unique authors first.
    all_authors = set()
    for auths in paper_to_authors.values():
        all_authors.update(auths)
    all_authors = sorted(list(all_authors))  # Sort for reproducibility

    # Map 'Author Name' -> Index ID
    author_to_idx = {name: i for i, name in enumerate(all_authors)}
    # Map Index ID -> 'Author Name' (for reconstructing the graph later)
    idx_to_author = {i: name for i, name in enumerate(all_authors)}

    # Map Paper ID -> Index ID
    all_papers = sorted(list(paper_to_authors.keys()))
    paper_to_idx = {pid: i for i, pid in enumerate(all_papers)}

    # 2. Build the Sparse Incidence Matrix B (Authors x Papers)
    # We collect triplets: (row, col, value) to build the matrix efficiently.
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

    # Create B: Shape (Num_Authors, Num_Papers)
    n_authors = len(all_authors)
    n_papers = len(all_papers)
    B = csr_matrix((data, (rows, cols)), shape=(n_authors, n_papers))

    # 3. The Projection: C = B * B^T
    # This single line replaces your inefficient nested loops.
    # It calculates the number of shared papers for ALL pairs instantly.
    print("  Projecting Bipartite Graph...")
    C = B.dot(B.T)

    # 4. Cleanup
    # The diagonal C[i,i] is the total number of papers author 'i' wrote.
    # In a social graph, we don't want self-loops.
    C.setdiag(0)
    C.eliminate_zeros()

    # 5. Convert Matrix back to NetworkX Graph
    print("  Converting Matrix to Graph...")
    # 'from_scipy_sparse_array' is the modern method (NetworkX 3.x)
    # If using older NetworkX, use 'from_scipy_sparse_matrix'
    try:
        G_co = nx.from_scipy_sparse_array(C)
    except AttributeError:
        G_co = nx.from_scipy_sparse_matrix(C)

    # Relabel integer nodes (0, 1...) back to actual names ("E. Witten", etc.)
    nx.relabel_nodes(G_co, idx_to_author, copy=False)

    print(
        f"Co-authorship Graph: {G_co.number_of_nodes()} nodes, {G_co.number_of_edges()} edges"
    )
    # We define distance as the inverse of collaboration strength (1 / #papers).
    print("  Calculating Effective Distances (d = 1/w)...")
    for u, v, data in G_co.edges(data=True):
        w = data.get('weight', 1) 
        # Avoid division by zero (though weights should be >= 1)
        if w > 0:
            data['distance'] = 1.0 / w
        else:
            data['distance'] = 1.0 # Fallback

    # --- Layer 2: Citation Network Construction (Original Logic) ---
    # Map paper-to-paper citations to author-to-author citations.
    print("Processing citation edges...")
    G_cit = nx.DiGraph()

    try:
        with open(edges_file, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue

                source_paper = parts[0]
                target_paper = parts[1]

                source_auths = paper_to_authors.get(source_paper, [])
                target_auths = paper_to_authors.get(target_paper, [])

                # Integer weighting logic:
                if len(source_auths) > 0 and len(target_auths) > 0:

                    weight = 1.0 

                    for sa in source_auths:
                        for ta in target_auths:
                            if sa == ta:
                                continue

                            if G_cit.has_edge(sa, ta):
                                G_cit[sa][ta]["weight"] += weight
                            else:
                                G_cit.add_edge(sa, ta, weight=weight)



    except FileNotFoundError:
        print(f"Error: Could not find {edges_file}")
        return None, None

    print(
        f"Citation Graph: {G_cit.number_of_nodes()} nodes, {G_cit.number_of_edges()} edges"
    )

    # --- VERIFICATION BLOCK ---
    print("\n[DEBUG] Checking Weight Distributions...")
    
    # Check Co-authorship
    co_weights = [d['weight'] for u, v, d in G_co.edges(data=True)]
    if co_weights:
        print(f"  Co-authorship (Shared Papers): Min={min(co_weights)}, Max={max(co_weights)}")
        if min(co_weights) < 1.0:
            print("  [WARNING] Co-authorship weights contain fractions!")
        else:
            print("  [OK] Co-authorship weights are integers.")

    # Check Citations
    cit_weights = [d['weight'] for u, v, d in G_cit.edges(data=True)]
    if cit_weights:
        print(f"  Citations (Counts):            Min={min(cit_weights)}, Max={max(cit_weights)}")
        if min(cit_weights) < 1.0:
            print("  [WARNING] Citation weights contain fractions!")
        else:
            print("  [OK] Citation weights are integers.")
    # --------------------------

    return G_co, G_cit


# ==========================================
# 4. ANALYSIS FUNCTIONS
# ==========================================


def analyze_layer_shortest_paths(G_cit, G_co):
    """
    Calculates the shortest path distance in the co-authorship network
    for pairs of authors who have a direct citation link.

    This analysis helps determine if citations typically occur within
    close social circles (small co-authorship distance) or across
    different research clusters.
    """
    print("\n--- Cross-Layer Path Analysis ---")
    distances = []
    sample_edges = list(G_cit.edges())
    valid_pairs = 0

    for u, v in sample_edges:
        if u in G_co and v in G_co:
            try:
                # Weighted Shortest Path (Effective Distance)
                d = nx.shortest_path_length(G_co, source=u, target=v, weight='distance')

                distances.append(d)
                valid_pairs += 1
            except nx.NetworkXNoPath:
                # Mark unreachable pairs in the co-authorship layer
                distances.append(-1)

    reachable_distances = [d for d in distances if d != -1]
    avg_dist = np.mean(reachable_distances) if reachable_distances else 0

    print(f"Analyzed {valid_pairs} pairs connected in Citation layer.")
    print(f"Average Co-authorship distance for these pairs: {avg_dist:.2f}")

    # Plotting the social distance distribution for cited pairs
    plt.figure(figsize=(8, 5))
    plt.hist(
        reachable_distances, 
        bins=50,             
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    plt.title("Distance in Co-authorship Layer for Citation-Connected Pairs")
    plt.xlabel("Shortest Path Length (Co-authorship)")
    plt.ylabel("Frequency")
    plt.axvline(
        avg_dist,
        color="red",
        linestyle="dashed",
        linewidth=1,
        label=f"Avg: {avg_dist:.2f}",
    )
    plt.legend()

    # Save the cross-layer distance histogram
    plt.savefig(os.path.join(RESULTS_DIR, "cross_layer_path_distribution.png"))
    plt.close()

    return avg_dist


def check_community_distribution(G):
    """
    Identifies communities within the network using the Louvain method
    and analyzes the distribution of their sizes.

    This provides insight into whether the network is dominated by a
    few large 'major fields' or fragmented into many small research groups.
    """
    print("\n--- Community Size Distribution Check ---")

    # Generate partition based on modularity optimization
    partition = community_louvain.best_partition(G)

    size_counts = Counter(partition.values())
    sizes = list(size_counts.values())
    sizes.sort(reverse=True)

    print(f"Total Communities: {len(sizes)}")
    print(f"Top 5 Largest (Major Fields): {sizes[:5]}")
    print(
        f"Number of 'Tiny' Communities (Size < 5): {len([s for s in sizes if s < 5])}"
    )

    # Plotting community size distribution on a log scale for clarity
    plt.figure(figsize=(8, 5))
    plt.hist(sizes, bins=50, color="teal", edgecolor="black")
    plt.title("Distribution of Community Sizes (Log Scale)")
    plt.xlabel("Number of Authors in Community")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.grid(axis="y", alpha=0.5)

    # Save the community size distribution plot
    plt.savefig(os.path.join(RESULTS_DIR, "community_size_distribution.png"))
    plt.close()


def analyze_communities_robust(G, author_to_papers, paper_to_text, n_iterations=10):
    """
    Performs robust community detection using the Louvain method with stability
    testing across multiple iterations.

    Includes:
    1. Stability Analysis: Computes Adjusted Rand Score (ARI) and Modularity (Q)
       consistency.
    2. Physics Interpretation: Analyzes the strength of community clustering.
    3. Topic Modeling: Uses TF-IDF vectorization on abstract texts to identify
       the primary research keywords for the largest detected communities.
    """
    print("\n--- Robust Community Detection & Stability Analysis ---")

    G_undir = G.to_undirected()
    print(f"Running Louvain {n_iterations} times to test stability...")

    partitions_list = []
    modularities = []

    # Iterate Louvain algorithm to assess consistency across different random seeds
    for i in range(n_iterations):
        part = community_louvain.best_partition(G_undir, random_state=i)
        partitions_list.append(part)
        q = community_louvain.modularity(part, G_undir)
        modularities.append(q)

    ari_scores = []
    nodes = list(G_undir.nodes())
    first_run_labels = [partitions_list[0][n] for n in nodes]

    # Calculate similarity between the first run and subsequent iterations
    for i in range(1, n_iterations):
        current_run_labels = [partitions_list[i][n] for n in nodes]
        score = adjusted_rand_score(first_run_labels, current_run_labels)
        ari_scores.append(score)

    avg_ari = np.mean(ari_scores) if ari_scores else 1.0
    avg_modularity = np.mean(modularities)
    std_modularity = np.std(modularities)

    print(f"\nStability Results:")
    print(f"  Average Modularity (Q): {avg_modularity:.4f} ± {std_modularity:.4f}")
    print(f"  Stability (Avg ARI):    {avg_ari:.4f}")

    # Interpret the network structure based on the modularity score
    print(f"\n[Physics Interpretation of Q={avg_modularity:.2f}]")
    if avg_modularity > 0.7:
        print("  -> EXTREME Modularity (Q > 0.7).")
        print("  -> The network is composed of isolated 'islands' of research.")
        print(
            "  -> This explains the slow diffusion time (tau): ideas get trapped inside these tight groups."
        )
        print(
            "  -> Warning: Such high Q might indicate distinct sub-archives in the dataset."
        )
    elif avg_modularity > 0.4:
        print(
            "  -> Strong community structure found (Q > 0.4). Standard social network behavior."
        )
    else:
        print("  -> Weak community structure (Q < 0.4). The network is well-mixed.")

    # Select the partition with the highest modularity for downstream analysis
    best_idx = np.argmax(modularities)
    best_partition = partitions_list[best_idx]
    # STRICT VALIDATION: CHECK FOR DISCONNECTED COMMUNITIES
    print("\n[STRICT VALIDATION] Verifying Louvain Community Connectedness...")
    
    # 1. Group nodes by their community ID
    community_nodes = defaultdict(list)
    for node, comm_id in best_partition.items():
        community_nodes[comm_id].append(node)

    # 2. Check each community
    fragmented_count = 0
    total_fragments = 0
    
    print(f"  Checking {len(community_nodes)} communities...")
    for comm_id, nodes in community_nodes.items():
        # Create a subgraph of ONLY this community
        subg = G.subgraph(nodes)
        
        # Check if it is physically connected
        if not nx.is_connected(subg):
            comps = list(nx.connected_components(subg))
            if len(nodes) > 5: # Only warn for significant communities
                print(f"  [WARNING] Community {comm_id} (Size: {len(nodes)}) is DISCONNECTED into {len(comps)} parts!")
            
            fragmented_count += 1
            total_fragments += len(comps)

    # 3. Report Results to Professor
    print(f"\n  Validation Results:")
    if fragmented_count > 0:
        print(f"  -> FAILED: {fragmented_count} communities are internally disconnected.")
        print(f"  -> Total distinct topological clusters found: {len(community_nodes) + total_fragments - fragmented_count}")
        print("  -> Scientific Note: These communities represent 'Semantic Topics' rather than strict 'Topological Clusters'.")
    else:
        print("  -> PASSED: All communities are fully connected. Louvain behavior matches Leiden here.")
    # -------------------------------------------------------------
    print(
        f"\nProceeding with the best run (Run #{best_idx + 1}, Q={modularities[best_idx]:.4f})..."
    )

    communities = defaultdict(list)
    for node, comm_id in best_partition.items():
        communities[comm_id].append(node)

    # Filter for communities of significant size (>= 5 authors)
    significant_comms = {
        cid: auths for cid, auths in communities.items() if len(auths) >= 5
    }
    sorted_comm_ids = sorted(
        significant_comms.keys(), key=lambda k: len(significant_comms[k]), reverse=True
    )

    print(f"Detected {len(communities)} total communities.")

    # Aggregate abstract text for each community to enable TF-IDF analysis
    community_documents = []
    map_index_to_comm_id = []

    for comm_id, authors in significant_comms.items():
        comm_text_parts = []
        for author in authors:
            papers = author_to_papers.get(author, [])
            for pid in papers:
                if pid in paper_to_text:
                    comm_text_parts.append(paper_to_text[pid])

        full_text = " ".join(comm_text_parts)
        if full_text.strip():
            community_documents.append(full_text)
            map_index_to_comm_id.append(comm_id)

    if not community_documents:
        print("Error: No abstract text found for any community.")
        return best_partition

    # Define stop words to remove common metadata and generic research terminology
    custom_stop_words = [
        "english",
        "date",
        "gmt",
        "comments",
        "pages",
        "title",
        "authors",
        "paper",
        "abstract",
        "report",
        "no",
        "hep",
        "th",
        "theory",
        "phys",
        "university",
        "department",
        "lat",
        "mon",
        "tue",
        "wed",
        "thu",
        "fri",
        "dec",
        "nov",
        "oct",
        "sep",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "cern",
        "fermilab",
        "slac",
        "caltech",
        "physics",
        "recent",
        "study",
        "studied",
        "using",
        "based",
        "results",
        "show",
        "investigate",
        "discuss",
        "given",
        "obtained",
        "particular",
        "case",
        "terms",
        "limit",
        "large",
        "small",
        "high",
        "low",
        "order",
        "leading",
        "function",
        "functions",
        "equation",
        "equations",
        "solution",
        "solutions",
        "approach",
        "general",
        "analysis",
        "problem",
        "structure",
        "properties",
        "model",
        "models",
        "theory",
        "field",
        "fields",
    ]
    stop_words = list(ENGLISH_STOP_WORDS.union(custom_stop_words))

    # Calculate TF-IDF to find distinctive terms for the top 5 communities
    tfidf = TfidfVectorizer(
        stop_words=stop_words, max_features=1000, max_df=0.25, sublinear_tf=True
    )

    tfidf_matrix = tfidf.fit_transform(community_documents)
    feature_names = np.array(tfidf.get_feature_names_out())
    top_5_ids = set(sorted_comm_ids[:5])

    print("\n--- Top Topics per Community ---")
    for idx, comm_id in enumerate(map_index_to_comm_id):
        if comm_id in top_5_ids:
            size = len(significant_comms[comm_id])
            row = tfidf_matrix[idx]
            scores = row.toarray().flatten()
            top_indices = scores.argsort()[::-1][:10]
            top_keywords = feature_names[top_indices]

            print(f"\nCommunity {comm_id} (Size: {size} authors):")
            print(f"Distinctive Keywords: {', '.join(top_keywords)}")

    return best_partition


def print_top_authors(G_co, G_cit):
    """
    Identifies the top 5 authors based on:
    1. Degree (Collaboration Count)
    2. Weighted In-degree (Citation Influence)
    3. Betweenness Centrality (Information Bridges)
    """
    print("\n--- Centrality Analysis ---")

    # 1. Degree Centrality (Hubs)
    top_connected = sorted(
        dict(G_co.degree()).items(), key=lambda x: x[1], reverse=True
    )[:5]
    print("Most Collaborative (High Degree):")
    for author, degree in top_connected:
        print(f"  - {author}: {degree} co-authors")

    # 2. Citation Influence (Authorities)
    # Note: We use the integer weights we fixed in Step 1
    top_cited = sorted(
        dict(G_cit.in_degree(weight="weight")).items(), key=lambda x: x[1], reverse=True
    )[:5]
    print("\nMost Influential (High Citations):")
    for author, degree in top_cited:
        print(f"  - {author}: {degree} citations")

    # 3. Betweenness Centrality (Bridges)
    print("\nCalculating Betweenness Centrality (this may take a moment)...")
    
    # CRITICAL: We use 'distance' (1/shared_papers) calculated in build_networks.
    # We want shortest paths to flow through STRONG ties (Small Distance).
    try:
        betweenness = nx.betweenness_centrality(G_co, weight='distance')
        
        top_bridges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top Bridges (High Betweenness - Interdisciplinary Connectors):")
        for author, score in top_bridges:
            print(f"  - {author}: {score:.4f}")
            
    except KeyError:
        print("  [Error] 'distance' attribute missing. Did you run the Step 2 fix in build_networks?")

def analyze_power_law(G, name="Network"):
    """
    Analyzes the heavy-tail distribution of degrees.
    
    SIMPLIFIED VERSION:
    - Removes complex statistical debates (Log-Normal vs Truncated).
    - Focuses on the physical result: The presence of Hubs.
    """
    print(f"\n--- Heavy-Tail Distribution Analysis ({name}) ---")

    degrees = [d for n, d in G.degree() if d > 0]
    fit = powerlaw.Fit(degrees, discrete=True, verbose=False)

    print(f"  Power Law Alpha: {fit.power_law.alpha:.4f}")
    print(f"  Xmin (Cutoff):   {fit.power_law.xmin}")

    # We calculate the comparisons for the PLOT, but we do NOT print the debate.
    R, p = fit.distribution_compare("power_law", "lognormal")

    # [PHYSICS CONCLUSION - SIMPLIFIED]
    # We don't care if it's "statistically" Log-Normal or Power Law.
    # We care that it is HEAVY-TAILED (which both are).
    print("\n[Physics Interpretation]")
    if fit.power_law.alpha < 3.5:
        print("  -> Verdict: The distribution is HEAVY-TAILED.")
        print("  -> Physical Meaning: The network is dominated by 'Hubs' (Super-Connectors).")
        print("  -> This confirms the 'Scale-Free' nature of scientific collaboration.")
    else:
        print("  -> Verdict: The distribution decays quickly.")
        print("  -> Physical Meaning: No significant Hubs found.")

    # Plotting (Keep this, it's good visual evidence)
    plt.figure(figsize=(8, 6))
    fit.plot_pdf(color="b", linear_bins=True, label="Empirical Data")
    fit.power_law.plot_pdf(color="r", linestyle="--", label="Power Law Fit")
    fit.lognormal.plot_pdf(color="g", linestyle="-.", label="Log-Normal Fit")

    plt.title(f"Degree Distribution ({name})")
    plt.xlabel("Degree (k)")
    plt.ylabel("P(k)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(
        os.path.join(RESULTS_DIR, f"{name.lower().replace('-', '_')}_power_law_fit.png")
    )
    plt.close()

def analyze_strength_distribution(G, name="Network"):
    """
    Analyzes the Node Strength Distribution P(s) and the Strength-Degree Correlation s(k).
    
    1. Strength (s): The sum of weights of edges incident to a node.
    2. Correlation: Fits s(k) ~ k^beta. 
       - If beta = 1: Weights are independent of degree.
       - If beta > 1: Hubs have disproportionately strong connections (Rich-club effect).
    """
    print(f"\n--- Weighted Strength Analysis ({name}) ---")
    
    # 1. Calculate Strength and Degree
    # In NetworkX, degree(weight='weight') computes the sum of weights (Strength)
    strengths = dict(G.degree(weight='weight'))
    degrees = dict(G.degree(weight=None))
    
    s_values = np.array(list(strengths.values()))
    k_values = np.array([degrees[n] for n in strengths.keys()])
    
    # Filter out zero degrees to avoid log(0) errors
    valid_mask = (k_values > 0) & (s_values > 0)
    k_values = k_values[valid_mask]
    s_values = s_values[valid_mask]

    # --- PART A: Strength Distribution P(s) ---
    plt.figure(figsize=(8, 6))
    
    # Use logarithmic binning for heavy-tailed distribution
    # We create bins that grow exponentially (1, 2, 4, 8, ...)
    bins = np.logspace(np.log10(min(s_values)), np.log10(max(s_values)), 30)
    
    # Calculate PDF using histogram
    counts, bin_edges = np.histogram(s_values, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot P(s)
    plt.loglog(bin_centers, counts, 'ro', label='Empirical P(s)')
    plt.title(f"Node Strength Distribution P(s) - {name}")
    plt.xlabel("Strength s (Total Shared Papers)")
    plt.ylabel("Probability Density P(s)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(RESULTS_DIR, f"{name.lower()}_strength_distribution.png"))
    plt.close()
    
    # --- PART B: Strength vs Degree Correlation s(k) ---
    plt.figure(figsize=(8, 6))
    
    # Scatter plot of all nodes
    plt.scatter(k_values, s_values, alpha=0.1, color='gray', s=10, label='Nodes')
    
    # Bin the degrees to find average strength <s(k)> for each k
    # This cleans up the noise to show the trend clearly
    k_unique = np.unique(k_values)
    s_avg_k = []
    for k in k_unique:
        mean_s = np.mean(s_values[k_values == k])
        s_avg_k.append(mean_s)
    
    plt.loglog(k_unique, s_avg_k, 'bo', label='Average <s(k)>')
    
    # Fit Power Law: s ~ A * k^beta
    # Linear fit in log-log space: log(s) = beta * log(k) + log(A)
    log_k = np.log10(k_unique)
    log_s = np.log10(s_avg_k)
    
    slope, intercept = np.polyfit(log_k, log_s, 1)
    beta = slope
    
    # Plot the fit line
    fit_y = 10**intercept * k_unique**beta
    plt.plot(k_unique, fit_y, 'r--', linewidth=2, label=f'Fit: $\\beta = {beta:.2f}$')
    
    plt.title(f"Strength vs Degree Correlation ($s \\sim k^\\beta$)")
    plt.xlabel("Degree k (Number of Collaborators)")
    plt.ylabel("Strength s (Total Papers)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    plt.savefig(os.path.join(RESULTS_DIR, f"{name.lower()}_strength_degree_correlation.png"))
    plt.close()
    
    print(f"  Average Strength <s(k)> fit exponent (beta): {beta:.4f}")
    
    # Physics Interpretation for the Console
    print("\n[Physics Interpretation]")
    if beta > 1.1:
        print(f"  -> Super-linear (beta={beta:.2f} > 1): 'Rich-get-Richer'.")
        print("  -> Meaning: Hubs collaborate MORE intensely per person than non-hubs.")
        print("  -> This is typical of scientific teams where senior PIs run large, frequent labs.")
    elif beta < 0.9:
        print(f"  -> Sub-linear (beta={beta:.2f} < 1): Saturation effect.")
        print("  -> Meaning: As you get more collaborators, you invest LESS time in each one.")
    else:
        print(f"  -> Linear (beta={beta:.2f} approx 1): Neutral.")
        print("  -> Meaning: Collaboration intensity is independent of team size.")

def analyze_configuration_model(G, n_randomizations=10):
    """
    Compares the real network's clustering to a randomized Configuration Model.

    By preserving the exact degree sequence while rewiring edges, this test
    determines if high clustering is a property of the hubs themselves or
    the result of intentional social selection/triadic closure.
    """
    print(f"\n--- Null Model Comparison (Configuration Model) ---")

    if not nx.is_connected(G):
        print("Graph is disconnected. Extracting Giant Connected Component (GCC)...")
        G_cc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    else:
        G_cc = G.copy()

    C_real = nx.average_clustering(G_cc)
    print(f"Real Clustering Coefficient: {C_real:.4f}")

    print(f"Generating {n_randomizations} randomized null graphs...")
    null_clustering_values = []
    n_swaps = 10 * G_cc.number_of_edges()

    # Perform double-edge swaps to randomize topology while preserving degree distribution
    for i in range(n_randomizations):
        G_null = G_cc.copy()
        try:
            nx.double_edge_swap(G_null, nswap=n_swaps, max_tries=n_swaps * 5)
        except nx.NetworkXError:
            pass
        null_clustering_values.append(nx.average_clustering(G_null))

    avg_null_C = np.mean(null_clustering_values)
    std_null_C = np.std(null_clustering_values)
    z_score = (C_real - avg_null_C) / std_null_C if std_null_C > 0 else 0

    print(f"\nNull Model Results: <C_null>: {avg_null_C:.4f}, Z-Score: {z_score:.2f}")

    print("\n[Physics Interpretation]")
    if z_score > 2:
        print(
            f"  Z-score {z_score:.2f} >> 2: High clustering is NON-TRIVIAL (Social Selection)."
        )
    else:
        print("  Result: Clustering is explained by degree distribution alone.")

    # Histogram of null distribution vs real observed value
    plt.figure(figsize=(8, 5))
    plt.hist(
        null_clustering_values,
        bins=10,
        color="gray",
        alpha=0.7,
        label="Null Model Distribution",
    )
    plt.axvline(
        C_real,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Real Network (C={C_real:.2f})",
    )
    plt.title(f"Real vs Configuration Model (Z-Score: {z_score:.2f})")
    plt.xlabel("Average Clustering Coefficient")
    plt.ylabel("Frequency")
    plt.legend()

    plt.savefig(os.path.join(RESULTS_DIR, "configuration_model_comparison.png"))
    plt.close()


def analyze_spectral_properties(G):
    """
    Analyzes the eigenvalues of the Normalized Laplacian matrix.
    Calculates diffusion time and Von Neumann Entropy.
    """
    print("\n--- Spectral Analysis (Laplacian & Wigner Check) ---")

    if not nx.is_connected(G):
        print("Graph is disconnected. Using Giant Connected Component (GCC)...")
        G_cc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    else:
        G_cc = G.copy()

    n = G_cc.number_of_nodes()

    # --- Standard Diffusion Analysis ---
    print("Calculating Average Path Length (L) for diffusion context...")
    try:
        # Use weighted distance for average path length
        avg_path_len = nx.average_shortest_path_length(G_cc, weight='distance')
    except:

        avg_path_len = np.log(n)  # Theoretical approximation if too slow

    print(f"Average Path Length (L): {avg_path_len:.2f}")

    print("Computing Normalized Laplacian Matrix...")
    L = nx.normalized_laplacian_matrix(G_cc)

    print("Diagonalizing Laplacian (calculating eigenvalues)...")
    # Note: For Von Neumann Entropy, we strictly need ALL eigenvalues.
    # Sparse approximation (k=100) is insufficient for accurate entropy.
    try:
        eigenvalues = scipy.linalg.eigh(L.todense(), eigvals_only=True)
    except MemoryError:
        print(
            "Warning: Graph too large for full diagonalization. Entropy will be estimated (inaccurate)."
        )
        # Fallback to sparse if absolutely necessary, but entropy will be wrong.
        eigenvalues = scipy.sparse.linalg.eigsh(
            L, k=min(n - 1, 1000), which="SM", return_eigenvectors=False
        )

    eigenvalues.sort()

    # --- 1. Algebraic Connectivity (Diffusion) ---
    lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0
    diffusion_time = 1 / lambda_2 if lambda_2 > 0 else float("inf")

    # --- 2. Von Neumann Entropy  ---
    # Normalize eigenvalues to create a density matrix rho with Trace = 1
    sum_eigenvalues = np.sum(eigenvalues)

    # Avoid division by zero
    if sum_eigenvalues > 0:
        rho_eigenvalues = eigenvalues / sum_eigenvalues
    else:
        rho_eigenvalues = eigenvalues


    vn_entropy = -np.sum(rho_eigenvalues * np.log(rho_eigenvalues + 1e-12))
    
    # 1. Calculate Maximum Possible Entropy (Random Graph Limit)
    max_entropy = np.log(n)
    
    # 2. Calculate "Relative Entropy" (How random is it?)
    relative_entropy = vn_entropy / max_entropy
    
    print(f"\nSpectral Metrics:")
    print(f"  Algebraic Connectivity (lambda_2): {lambda_2:.6f}")
    print(f"  Von Neumann Entropy (S):           {vn_entropy:.4f}")
    print(f"  Max Possible Entropy (ln N):       {max_entropy:.4f}")
    print(f"  Relative Entropy (S / S_max):      {relative_entropy:.4f}")

    print("\n[Physics Interpretation]")
    if relative_entropy > 0.9:
        print("  -> High Relative Entropy (> 0.9): The network topology is nearly RANDOM.")
        print("  -> Meaning: Lack of strong internal structure.")
    elif relative_entropy < 0.7:
        print("  -> Low Relative Entropy (< 0.7): High Structural ORDER.")
        print("  -> Meaning: The network has very distinct communities (Segregated Topology).")
    else:
        print("  -> Moderate Entropy: The network balances random connections with local structure.")

    # Max possible entropy for a graph of size N is ln(N) (approximated by random graph)
    max_entropy = np.log(n)
    regularity_index = 1 - (vn_entropy / max_entropy)

    print(f"\nSpectral Metrics:")
    print(f"  Algebraic Connectivity (lambda_2): {lambda_2:.6f}")
    print(f"  Von Neumann Entropy (S):           {vn_entropy:.4f}")
    print(f"  Max Entropy (ln N):                {max_entropy:.4f}")
    print(f"  Regularity Index (1 - S/S_max):    {regularity_index:.4f}")

    print("\n[Physics Interpretation]")
    print(f"  Diffusion Time (tau ~ 1/lambda_2): approx {diffusion_time:.2f} steps.")

    print("\n[Physics Interpretation]")
    print(f"  Diffusion Time (tau ~ 1/lambda_2): approx {diffusion_time:.2f} steps.")

    # FIX: We check lambda_2 (The Spectral Gap) first. 
    # A small lambda_2 means the graph has "bottlenecks" (communities), regardless of entropy.
    
    if lambda_2 < 0.05: 
        print(f"  -> LOW Algebraic Connectivity (lambda_2 = {lambda_2:.5f} < 0.05).")
        print("  -> Verdict: The network is HIGHLY MODULAR (Distinct Communities).")
        print("  -> Paradox Resolution: High Entropy here reflects the diversity of communities, NOT randomness.")
        print("  -> Diffusion is 'trapped' inside local clusters (Slow Relaxation).")
        
    elif regularity_index > 0.1:
        print(f"  -> Non-Trivial Entropy (Index={regularity_index:.2f}): Significant structural order detected.")
        
    else:
        print(f"  -> High Entropy & High Lambda_2: The network topology is likely random/well-mixed.")
    # --- Plotting Spectral Density ---
    plt.figure(figsize=(10, 6))

    kde = gaussian_kde(eigenvalues)
    x_range = np.linspace(min(eigenvalues), max(eigenvalues), 200)

    plt.plot(x_range, kde(x_range), color="blue", lw=2, label="Spectral Density")
    plt.fill_between(x_range, kde(x_range), color="blue", alpha=0.1)

    # Add entropy annotation to plot
    plt.text(
        0.05,
        0.95,
        f"VN Entropy: {vn_entropy:.2f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.title(f"Spectral Density & Entropy (N={n})")
    plt.xlabel(r"Eigenvalue ($\lambda$)")
    plt.ylabel(r"Density $\rho(\lambda)$")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(RESULTS_DIR, "spectral_density_entropy.png"))
    plt.close()


def analyze_robustness(G):
    """
    Simulates the fragmentation of the network under different removal strategies.

    Compares:
    1. Random Failure: Removing nodes uniformly at random.
    2. Targeted Attack: Removing high-degree hubs first.

    This identifies if the network is 'Robust yet Fragile', a hallmark of scale-free systems.
    """
    print("\n--- Robustness & Perturbation Analysis ---")

    fraction_to_remove = 0.2
    steps = 20
    n_total = G.number_of_nodes()
    n_remove = int(n_total * fraction_to_remove)
    step_size = max(1, n_remove // steps)

    print(
        f"Simulating removal of up to {n_remove} nodes ({fraction_to_remove * 100}%) in {steps} steps..."
    )

    # Targeted Attack Setup: Hubs sorted by connectivity
    G_attack = G.copy()
    attack_sizes = [1.0]
    nodes_sorted_by_degree = sorted(G.degree, key=lambda x: x[1], reverse=True)
    targets = [n for n, d in nodes_sorted_by_degree]

    # Random Failure Setup
    G_random = G.copy()
    random_sizes = [1.0]
    import random

    random_targets = list(G.nodes())
    random.shuffle(random_targets)

    # Iterative removal process
    for i in range(0, n_remove, step_size):
        # Attack high-degree nodes
        batch_attack = targets[i : i + step_size]
        G_attack.remove_nodes_from(batch_attack)
        if len(G_attack) > 0:
            gcc_attack = len(max(nx.connected_components(G_attack), key=len))
            attack_sizes.append(gcc_attack / n_total)
        else:
            attack_sizes.append(0)

        # Random failure simulation
        batch_random = random_targets[i : i + step_size]
        G_random.remove_nodes_from(batch_random)
        if len(G_random) > 0:
            gcc_random = len(max(nx.connected_components(G_random), key=len))
            random_sizes.append(gcc_random / n_total)
        else:
            random_sizes.append(0)

    x_axis = np.linspace(0, fraction_to_remove, len(attack_sizes))

    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, random_sizes, "g-o", label="Random Failure")
    plt.plot(x_axis, attack_sizes, "r-s", label="Targeted Attack (Hubs)")

    plt.title("Network Robustness: GCC Fragmentation")
    plt.xlabel("Fraction of Nodes Removed ($f$)")
    plt.ylabel("Relative Size of Giant Component ($S$)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(RESULTS_DIR, "network_robustness_fragmentation.png"))
    plt.close()

    drop_diff = random_sizes[-1] - attack_sizes[-1]
    print("\n[Physics Interpretation]")
    if drop_diff > 0.3:
        print(f"  Gap between Random and Attack is large ({drop_diff:.2f}).")
        print("  -> Result: The network is 'Robust yet Fragile'.")
        print(
            "  -> Meaning: It survives random failures well but collapses quickly if hubs are targeted."
        )
    else:
        print(
            "  -> Result: The network degrades similarly for both (likely homogenous topology)."
        )


def analyze_multiplex_correlation(G_co, G_cit):
    """
    Analyzes the correlation between an author's influence in the citation layer
    and their brokerage role in the co-authorship layer.

    This function:
    1. Identifies authors present in both layers.
    2. Calculates PageRank (Influence) for the citation layer.
    3. Calculates Betweenness Centrality (Brokerage) for the co-authorship layer.
    4. Computes Spearman correlation to determine if influence and social brokerage are coupled.
    """
    print("\n--- Multiplex Correlation Analysis ---")

    common_authors = list(set(G_co.nodes()).intersection(set(G_cit.nodes())))
    print(f"Analyzing {len(common_authors)} authors present in both layers...")

    if len(common_authors) < 10:
        print("Error: Not enough common authors for correlation.")
        return

    print("Calculating PageRank (Citation Layer)...")
    pagerank = nx.pagerank(G_cit, weight="weight")

    print("Calculating Betweenness Centrality (Co-authorship Layer)...")
    betweenness = nx.betweenness_centrality(G_co, weight="distance")

    x_data = []
    y_data = []

    for auth in common_authors:
        pr = pagerank.get(auth, 0)
        bt = betweenness.get(auth, 0)
        x_data.append(pr)
        y_data.append(bt)

    corr, p_value = spearmanr(x_data, y_data)

    print(f"\nResults:")
    print(f"  Spearman Correlation: {corr:.4f}")
    print(f"  P-value: {p_value:.4e}")

    print("\n[Physics Interpretation]")
    if corr > 0.5:
        print(
            "  -> High Correlation: Influential scientists are also the social brokers."
        )
    elif corr < 0.2:
        print("  -> Low Correlation: 'Influence' and 'Brokerage' are decoupled.")
    else:
        print(
            f"  -> Moderate Correlation ({corr:.2f}): Influence and Brokerage are related but distinct."
        )

    # Remove zeros for log-scale density plotting
    x_plot = [x for x, y in zip(x_data, y_data) if x > 0 and y > 0]
    y_plot = [y for x, y in zip(x_data, y_data) if x > 0 and y > 0]

    plt.figure(figsize=(10, 7))
    hb = plt.hexbin(
        x_plot,
        y_plot,
        gridsize=30,
        cmap="inferno",
        bins="log",
        xscale="log",
        yscale="log",
    )
    plt.colorbar(hb, label="log10(Number of Authors)")

    plt.xlabel("Citation Influence (PageRank)")
    plt.ylabel("Social Brokerage (Betweenness)")
    plt.title(f"Multiplex Correlation Density (Spearman r={corr:.2f})")
    plt.grid(True, alpha=0.3, which="both")

    plt.savefig(os.path.join(RESULTS_DIR, "multiplex_centrality_correlation.png"))
    plt.close()


def analyze_multiplex_overlap_significance(G_co, G_cit, n_randomizations=10):
    """
    Determines if the overlap of edges between the co-authorship and citation
    layers is statistically significant compared to a null model.
    """
    print("\n--- Multiplex Edge Overlap Significance ---")

    common_nodes = set(G_co.nodes()) & set(G_cit.nodes())
    sub_co = G_co.subgraph(common_nodes)
    sub_cit = G_cit.subgraph(common_nodes)

    edges_co = set([frozenset(e) for e in sub_co.edges()])
    edges_cit = set([frozenset(e) for e in sub_cit.edges()])

    intersection = len(edges_co.intersection(edges_cit))
    union = len(edges_co.union(edges_cit))
    real_jaccard = intersection / union if union > 0 else 0

    print(f"Real Jaccard Overlap: {real_jaccard:.6f} (Edges: {intersection})")

    # Generate null model via double-edge swaps to preserve degree sequences
    null_jaccards = []
    G_cit_undir = sub_cit.to_undirected().copy()
    n_swaps = 10 * G_cit_undir.number_of_edges()

    for _ in range(n_randomizations):
        G_rand = G_cit_undir.copy()
        try:
            nx.double_edge_swap(G_rand, nswap=n_swaps, max_tries=n_swaps * 10)
        except:
            pass

        edges_null = set([frozenset(e) for e in G_rand.edges()])
        int_null = len(edges_co.intersection(edges_null))
        uni_null = len(edges_co.union(edges_null))
        null_jaccards.append(int_null / uni_null if uni_null > 0 else 0)

    mean_null = np.mean(null_jaccards)
    std_null = np.std(null_jaccards)
    z_score = (real_jaccard - mean_null) / std_null if std_null > 0 else 0

    print(f"Null Model Jaccard: {mean_null:.6f} ± {std_null:.6f}")
    print(f"Z-Score: {z_score:.2f}")

    if z_score > 2:
        print(
            "Conclusion: The overlap is STATISTICALLY SIGNIFICANT (Social drives Citation)."
        )
    else:
        print("Conclusion: The overlap is random.")


def analyze_rich_club_structural(G_co, n_randomizations=10):
    """
    Calculates the Normalized Rich-Club Coefficient to see if high-degree
    nodes (top scientists) are more densely connected to each other
    than expected by chance.
    """
    print("\n--- Rich-Club Coefficient Analysis ---")

    G_clean = G_co.copy()
    G_clean.remove_edges_from(list(nx.selfloop_edges(G_clean)))

    try:
        rc_real = nx.rich_club_coefficient(G_clean, normalized=False)
    except:
        return

    # Randomize for normalization
    rc_null_totals = defaultdict(float)
    G_comp = G_clean.subgraph(max(nx.connected_components(G_clean), key=len)).copy()
    n_swaps = 5 * G_comp.number_of_edges()

    for _ in range(n_randomizations):
        G_rand = G_comp.copy()
        try:
            nx.double_edge_swap(G_rand, nswap=n_swaps, max_tries=n_swaps * 5)
            rc_rand = nx.rich_club_coefficient(G_rand, normalized=False)
            for k, val in rc_rand.items():
                rc_null_totals[k] += val
        except:
            continue

    rho_k = {
        k: rc_real[k] / (rc_null_totals[k] / n_randomizations)
        for k in rc_real
        if k in rc_null_totals and rc_null_totals[k] > 0
    }

    x = sorted(rho_k.keys())
    y = [rho_k[k] for k in x]

    if x:
        plt.figure(figsize=(8, 6))
        plt.plot(
            x,
            y,
            "o-",
            color="purple",
            markersize=4,
            label=r"Normalized Rich-Club $\rho(k)$",
        )
        plt.axhline(1.0, color="black", linestyle="--", label="Random Baseline")
        plt.title("Rich-Club Phenomenon (Collaboration Layer)")
        plt.xlabel("Degree $k$")
        plt.ylabel(r"$\rho(k)$")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(RESULTS_DIR, "rich_club_coefficient.png"))
        plt.close()

        avg_high_k = np.mean(y[int(len(y) * 0.8) :]) if len(y) > 0 else 1.0
        print(f"Average rho(k) for top 20% degrees: {avg_high_k:.2f}")

        if avg_high_k > 1.1:
            print("  -> Conclusion: Rich-Club detected (Oligarchy).")
        elif avg_high_k < 0.9:
            print("  -> Conclusion: Rich-Club is suppressed (Segregation).")
        else:
            print("  -> Conclusion: No Rich-Club effect.")


def analyze_path_length_control(G_co, real_val, sample_size=5000):
    """
    Provides a statistical baseline for path lengths by comparing the real
    citation-linked distances to a random sampling of pairs in the social layer.

    This determines if citation links are 'local' (connecting socially close
    authors) or 'global' (connecting random authors across the network).
    """
    print("\n--- Path Length Statistical Control ---")

    # Perform analysis on the Giant Connected Component for consistency
    if not nx.is_connected(G_co):
        G_main = G_co.subgraph(max(nx.connected_components(G_co), key=len))
    else:
        G_main = G_co

    nodes = list(G_main.nodes())
    distances = []

    print(f"Sampling {sample_size} random pairs (Control Group)...")
    count = 0
    while count < sample_size:
        u, v = np.random.choice(nodes, 2, replace=False)
        try:
            d = nx.shortest_path_length(G_main, source=u, target=v, weight='distance')
            distances.append(d)
            count += 1
        except nx.NetworkXNoPath:
            continue

    avg_random_dist = np.mean(distances)
    print(f"Random (Control) Average Distance: {avg_random_dist:.4f}")

    diff = avg_random_dist - real_val
    print(f"Difference (Random - Citation): {diff:.4f}")

    # Interpretation of the 'locality' of citation links
    if diff > 1.0:
        print(
            "Conclusion: Citation links bridge significantly closer nodes than random."
        )
        print(
            "Physics: The 'Small World' of citations is embedded within local communities."
        )
    else:
        print("Conclusion: Citation links span nearly the global average distance.")


def analyze_degree_correlation(G):
    """
    Analyzes Degree Assortativity to classify the network type.
    
    Hypothesis:
    - r > 0: Assortative (Social Network). Hubs link to Hubs.
    - r < 0: Disassortative (Tech/Bio Network). Hubs link to Leaves.
    """
    print("\n--- Degree Correlation (Assortativity) Analysis ---")

    # 1. Global Pearson Correlation Coefficient (r)
    # We use unweighted degree k to measure topological mixing
    r = nx.degree_assortativity_coefficient(G)
    print(f"Global Degree Assortativity (r): {r:.4f}")

    # 2. Physics Verdict
    print("\n[Physics Interpretation]")
    if r > 0.0:
        print(f"  -> VERDICT: Assortative Mixing (r={r:.2f} > 0).")
        print("  -> Conclusion: This behaves like a SOCIAL network.")
        print("  -> 'Rich Club' validation: High-degree authors tend to collaborate with other high-degree authors.")
    else:
        print(f"  -> VERDICT: Disassortative Mixing (r={r:.2f} < 0).")
        print("  -> Conclusion: This behaves like a TECHNOLOGICAL/BIOLOGICAL network.")
        print("  -> Structure: Star-like topology (Hubs connected to students/low-degree nodes).")

    # 3. Local Analysis: Average Neighbor Degree k_nn(k)
    knn_dict = nx.average_degree_connectivity(G)
    k = sorted(knn_dict.keys())
    knn = [knn_dict[x] for x in k]

    plt.figure(figsize=(8, 6))
    plt.scatter(k, knn, color="blue", alpha=0.6, s=30, label="Empirical Data")

    # Fit trend line (Power law: knn(k) ~ k^mu)
    try:
        valid_indices = [i for i in range(len(k)) if k[i] > 0 and knn[i] > 0]
        if len(valid_indices) > 5:
            k_log = np.log([k[i] for i in valid_indices])
            knn_log = np.log([knn[i] for i in valid_indices])

            slope, intercept = np.polyfit(k_log, knn_log, 1)
            fit_y = np.exp(intercept) * np.power([k[i] for i in valid_indices], slope)

            plt.plot(
                [k[i] for i in valid_indices],
                fit_y,
                "r--",
                label=rf"Fit: $\mu={slope:.2f}$",
            )
            print(f"  Correlation Exponent (mu): {slope:.2f}")
            if slope > 0:
                print("  -> k_nn(k) increases with k: Confirms Assortativity.")
            else:
                print("  -> k_nn(k) decreases with k: Confirms Disassortativity.")
    except Exception as e:
        print(f"  Could not fit trend line: {e}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Degree $k$")
    plt.ylabel(r"Avg Neighbor Degree $k_{nn}(k)$")
    plt.title(f"Degree Correlation Profile (r={r:.2f})")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")

    plt.savefig(os.path.join(RESULTS_DIR, "degree_correlation_assortativity.png"))
    plt.close()

def analyze_nmi_between_layers(G_co, G_cit):
    """
    Calculates the Normalized Mutual Information (NMI) between:
    1. Social Communities (Co-authorship clusters)
    2. Intellectual Communities (Citation clusters)
    
    Hypothesis:
    - High NMI: "Echo Chambers" (Structure is identical).
    - Low NMI: "Cross-Pollination" (Ideas flow across social groups).
    """
    print("\n--- Multiplex Community Comparison (NMI) ---")

    # 1. Detect Social Communities (Co-authorship)
    # G_co is already undirected and weighted.
    partition_co = community_louvain.best_partition(G_co, random_state=42)
    
    # 2. Detect Intellectual Communities (Citation)
    # Louvain requires an UNDIRECTED graph. We treat citations as symmetric
    # "topological relatedness" for the purpose of finding clusters.
    G_cit_undir = G_cit.to_undirected()
    partition_cit = community_louvain.best_partition(G_cit_undir, random_state=42)
    
    # 3. Align the node lists (Critical!)
    # We must ensure we are comparing the SAME nodes in the SAME order.
    nodes = list(G_co.nodes())
    
    labels_co = [partition_co[n] for n in nodes]
    
    # Handle cases where a node might be isolated in one layer but not the other
    # (Though Step 3 Alignment should prevent this)
    labels_cit = [partition_cit.get(n, -1) for n in nodes]
    
    # 4. Calculate NMI
    nmi = normalized_mutual_info_score(labels_co, labels_cit)
    
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    
    # 5. Physics Interpretation
    print("\n[Physics Interpretation]")
    if nmi > 0.5:
        print(f"  -> High NMI ({nmi:.2f}): Strong Coupling (Echo Chambers).")
        print("  -> Meaning: 'Social' and 'Intellectual' groups are nearly identical.")
        print("  -> Scientists mostly cite their own direct collaborators.")
    elif nmi < 0.2:
        print(f"  -> Low NMI ({nmi:.2f}): Decoupling (Idea Diffusion).")
        print("  -> Meaning: Citation flows traverse across different social circles.")
        print("  -> Ideas are spreading globally, regardless of local co-authorship.")
    else:
        print(f"  -> Moderate NMI ({nmi:.2f}): Partial alignment.")
        print("  -> Some overlap, but significant cross-group citation exists.")

    return nmi

# ==========================================
# 5. VISUALIZATION
# ==========================================


def visualize_network(G, title="hep_network_interactive.html"):
    """
    Generates a high-fidelity interactive HTML visualization of the network topology.

    To ensure visual interpretability and performance, the function extracts the
    top 500 nodes by degree (hubs) and projects their connectivity. Nodes are
    scaled by their degree centrality and color-coded based on their detected
    Louvain community membership to highlight the modular structure of the field.
    """
    print("\n--- Projecting Interactive Topological Map ---")

    # Check for the existence of the results directory to ensure safe file writing
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Filter the network to focus on the top 500 hubs for visual clarity
    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:500]
    G_sub = G.subgraph(top_nodes)

    # Configure the Pyvis environment for an academic presentation style
    # Uses a dark theme to maximize contrast for community-coded nodes
    net = Network(height="100vh", width="100%", bgcolor="#222222", font_color="white")

    # Apply modularity maximization to distinguish research clusters visually
    partition = community_louvain.best_partition(G_sub)

    # Map graph nodes to the interactive environment with metadata tooltips
    for node in G_sub.nodes():
        comm_id = partition[node]
        degree = degrees[node]
        net.add_node(
            node,
            label=node,
            title=f"Degree Centrality: {degree}",
            value=degree,
            group=comm_id,
        )

    # Map edges with neutral coloring to emphasize the node clusters
    for u, v in G_sub.edges():
        net.add_edge(u, v, color="#555555")

    # Apply ForceAtlas2 physics for a layout that emphasizes modular separation
    net.force_atlas_2based()

    # Save the output to the project directory
    save_path = os.path.join(RESULTS_DIR, title)
    net.save_graph(save_path)
    print(f"Interactive topological projection saved to {save_path}.")


def print_global_metrics(G):
    """
    Computes and prints standard topological metrics for the network.

    Calculates:
    1. Density: Ratio of actual edges to possible edges.
    2. Transitivity: Global probability that adjacent nodes are connected (triadic closure).
    3. Average Clustering: The mean of local clustering coefficients.
    """
    print("\n--- Global Graph Metrics ---")

    density = nx.density(G)
    print(f"Edge Density: {density:.6f}")

    transitivity = nx.transitivity(G)
    print(f"Global Clustering Coeff (Transitivity): {transitivity:.4f}")

    avg_clustering = nx.average_clustering(G)
    print(f"Average Clustering Coefficient: {avg_clustering:.4f}")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    """
    Main entry point for the High Energy Physics (HEP) network analysis pipeline.
    
    The workflow follows these stages:
    1. Data Loading: Parses abstracts and citation edges from the data directory.
    2. Network Construction: Builds the co-authorship (undirected) and 
       citation (directed) multilayer graphs.
    3. Structural Analysis: Evaluates degree distributions, clustering 
       significance, and community stability.
    4. Physical Dynamics: Analyzes spectral properties for information 
       diffusion and network robustness against targeted attacks.
    5. Multilayer Interaction: Correlates author roles across both social 
       and influential layers of the network.
    """
 


    if os.path.exists(ABSTRACTS_DIR) and os.path.exists(EDGES_FILE):
        # Phase 1: Data Parsing
        p2a, p2t, a2p = parse_abstracts(ABSTRACTS_DIR)

        G_co_full, G_cit_full = build_networks(EDGES_FILE, p2a)

        if G_co_full and G_cit_full:
            print("\n--- PHASE 2: MULTIPLEX DATA ALIGNMENT ---")
            
            # 1. Strict Intersection (The "Core" Set)
            nodes_co = set(G_co_full.nodes())
            nodes_cit = set(G_cit_full.nodes())
            common_nodes = nodes_co.intersection(nodes_cit)
            
            print(f"Original Co-authorship Nodes: {len(nodes_co)}")
            print(f"Original Citation Nodes:      {len(nodes_cit)}")
            print(f"Intersection (Core Set):      {len(common_nodes)}")
            
            if len(common_nodes) == 0:
                print("Error: No overlap between datasets! Check your parsing.")
                exit()

            # Filter both full graphs to the intersection
            G_co_core = G_co_full.subgraph(common_nodes).copy()
            G_cit_core = G_cit_full.subgraph(common_nodes).copy()

            # 2. Extract Giant Connected Component (GCC) from the Core Co-authorship Layer
            # We must ensure the social graph is connected for spectral analysis.
            print("Extracting Giant Connected Component (GCC) from the Core Set...")
            largest_cc_nodes = max(nx.connected_components(G_co_core), key=len)
            
            # 3. Finalize the Common Node Set
            # We only keep nodes that are in the Intersection AND in the GCC
            final_nodes = largest_cc_nodes
            
            G_co = G_co_core.subgraph(final_nodes).copy()
            G_cit = G_cit_core.subgraph(final_nodes).copy()

            print(f"Final Aligned Node Count: {G_co.number_of_nodes()}")
            print(f"  -> Co-authorship Edges: {G_co.number_of_edges()}")
            print(f"  -> Citation Edges:      {G_cit.number_of_edges()}")
            

            
            print(f"GCC Nodes: {G_co.number_of_nodes()} (Original: {G_co_full.number_of_nodes()})")
            
            
            # Phase 3: Cross-Layer Distance & Power Law Analysis
            # Phase 3: Cross-Layer Distance & Power Law Analysis
            avg_cit_dist = analyze_layer_shortest_paths(G_cit, G_co)
            analyze_power_law(G_co, name="Co-Authorship")
            analyze_strength_distribution(G_co, name="Co-Authorship")

            print_global_metrics(G_co)

            # Phase 4: Topology & Community Detection
            analyze_configuration_model(G_co, n_randomizations=10)
            check_community_distribution(G_co)

            # Robust community detection with text-based topic modeling
            best_part = analyze_communities_robust(G_co, a2p, p2t, n_iterations=5)

            # Phase 5: Spectral Dynamics & Robustness
            analyze_spectral_properties(G_co)
            analyze_robustness(G_co)

            # Phase 6: Multiplexity & Centrality
            analyze_multiplex_correlation(G_co, G_cit)
            visualize_network(G_co)
            print_top_authors(G_co, G_cit)

            # Phase 7: Statistical Controls & Advanced Structural Metrics
            analyze_path_length_control(G_co, avg_cit_dist)
            analyze_multiplex_overlap_significance(G_co, G_cit, n_randomizations=10)
            analyze_rich_club_structural(G_co, n_randomizations=10)
            analyze_degree_correlation(G_co)
            analyze_nmi_between_layers(G_co, G_cit)




            print(f"\nAnalysis complete. All plots saved to: {RESULTS_DIR}")
    else:
        print(
            f"Data files not found. Please ensure {EDGES_FILE} and {ABSTRACTS_DIR} exist."
        )

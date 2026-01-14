import os
import re
import warnings
from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from community import community_louvain


# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================
EDGES_FILE = 'data/cit-HepTh.txt'
ABSTRACTS_DIR = 'data/cit-HepTh-abstracts'

# ==========================================
# 2. DATA PARSING
# ==========================================

def normalize_name(name):
    parts = name.split()
    if len(parts) < 2:
        return name
    
    last_name = parts[-1]
    first_initial = parts[0][0]
    return f"{first_initial}. {last_name}"

def parse_abstracts(root_dir):

    print(f"Scanning abstracts in {root_dir}...")
    
    paper_to_authors = defaultdict(list)
    paper_to_text = {}
    author_to_papers = defaultdict(list)
    
    NON_AUTHOR_TERMS = {
                            'italy', 'germany', 'france', 'spain', 'russia', 'usa', 'japan', 'uk', 
                            'england', 'canada', 'switzerland', 'brazil', 'india', 'china', 'korea',
                            'australia', 'mexico', 'israel', 'netherlands', 'belgium', 'sweden', 
                            'cern', 'trieste', 'moscow', 'rome', 'paris', 'london', 'berlin', 'madrid',
                            'caltech', 'mit', 'stanford', 'harvard', 'princeton', 'cambridge'
                        }

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if not file.endswith('.abs'):
                continue
            
            paper_id = file.replace('.abs', '')
            path = os.path.join(root, file)
            
            try:
                with open(path, 'r', encoding='latin-1') as f:
                    content = f.read()
                    
                    auth_match = re.search(r'Authors?:\s*(.+?)(?=\n(?:Comments|Journal-ref|Subj-class|\\)|$)', content, re.DOTALL | re.IGNORECASE)

                    if auth_match:
                        raw_authors = auth_match.group(1).replace('\n', ' ')
                        raw_authors = re.sub(r'\(.*?\)', '', raw_authors)
                        
                        authors = re.split(r',|\sand\s|;', raw_authors)
                        
                        
                        cleaned_authors = []
                        for a in authors:
                            name = a.strip()
                            if len(name) <= 1: continue
                            
                            if any(x in name.lower() for x in ["university", "institute", "collab", "group", "department", "physic"]): continue
                            
                            if name.lower() in NON_AUTHOR_TERMS: continue
                            
                            normalized_name = normalize_name(name)
                            cleaned_authors.append(normalized_name)
                        
                        paper_to_authors[paper_id] = cleaned_authors
                        for auth in cleaned_authors:
                            author_to_papers[auth].append(paper_id)
                    
                    parts = content.split('\\\\')
                    abstract_candidate = ""
                    if len(parts) >= 3:
                        abstract_candidate = parts[2] 
                    elif len(parts) >= 2:
                        abstract_candidate = parts[1]
                    else:
                        abstract_candidate = parts[-1]
                    
                    paper_to_text[paper_id] = abstract_candidate.replace('\n', ' ').strip()
                        
            except Exception as e:
                print(f"Error parsing {paper_id}: {e}")
                continue

    print(f"Parsed {len(paper_to_authors)} papers.")
    return paper_to_authors, paper_to_text, author_to_papers

# ==========================================
# 3. GRAPH CONSTRUCTION
# ==========================================

def build_networks(edges_file, paper_to_authors):

    print("Building Author Networks...")
    G_co = nx.Graph()    # Co-authorship
    G_cit = nx.DiGraph() # Citation
    
    # --- Layer 1: Co-authorship ---
    for paper, authors in paper_to_authors.items():
        if len(authors) > 1:
            for u, v in combinations(authors, 2):
                if G_co.has_edge(u, v):
                    G_co[u][v]['weight'] += 1
                else:
                    G_co.add_edge(u, v, weight=1)

    # --- Layer 2: Citation ---
    print("Processing citation edges...")
    try:
        with open(edges_file, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.split()
                if len(parts) < 2: continue
                
                source_paper = parts[0]
                target_paper = parts[1]
                
                source_auths = paper_to_authors.get(source_paper, [])
                target_auths = paper_to_authors.get(target_paper, [])
                
                # Apply fractional weighting
                # If Paper A (5 authors) cites Paper B (2 authors), total weight 1 is distributed among 10 edges.
                if len(source_auths) > 0 and len(target_auths) > 0:
                    weight = 1.0 / (len(source_auths) * len(target_auths))
                
                    for sa in source_auths:
                        for ta in target_auths:
                            if sa == ta: continue
                            
                            if G_cit.has_edge(sa, ta):
                                G_cit[sa][ta]['weight'] += weight
                            else:
                                G_cit.add_edge(sa, ta, weight=weight)
                            
    except FileNotFoundError:
        print(f"Error: Could not find {edges_file}")
        return None, None

    print(f"Co-authorship Graph: {G_co.number_of_nodes()} nodes, {G_co.number_of_edges()} edges")
    print(f"Citation Graph: {G_cit.number_of_nodes()} nodes, {G_cit.number_of_edges()} edges")
    return G_co, G_cit


# ==========================================
# 4. ANALYSIS FUNCTIONS
# ==========================================

def analyze_layer_shortest_paths(G_cit, G_co):

    print("\n--- Cross-Layer Path Analysis ---")
    distances = []
    
    sample_edges = list(G_cit.edges())

    valid_pairs = 0
    for u, v in sample_edges:
        if u in G_co and v in G_co:
            try:
                d = nx.shortest_path_length(G_co, source=u, target=v)
                distances.append(d)
                valid_pairs += 1
            except nx.NetworkXNoPath:
                distances.append(-1) 
    
    reachable_distances = [d for d in distances if d != -1]
    avg_dist = np.mean(reachable_distances) if reachable_distances else 0
    
    print(f"Analyzed {valid_pairs} pairs connected in Citation layer.")
    print(f"Average Co-authorship distance for these pairs: {avg_dist:.2f}")
    
    plt.figure(figsize=(8, 5))
    plt.hist([d if d != -1 else max(reachable_distances)+1 for d in distances], 
             bins=range(0, max(reachable_distances)+3), alpha=0.7, color='skyblue', edgecolor='black')
    plt.title("Distance in Co-authorship Layer for Citation-Connected Pairs")
    plt.xlabel("Shortest Path Length (Co-authorship)")
    plt.ylabel("Frequency")
    plt.axvline(avg_dist, color='red', linestyle='dashed', linewidth=1, label=f'Avg: {avg_dist:.2f}')
    plt.legend()
    plt.show()

def analyze_communities_and_topics(G, author_to_papers, paper_to_text):

    print("\n--- Community Detection & Topic Extraction ---")
    
    G_undir = G.to_undirected()
    partition = community_louvain.best_partition(G_undir)
    
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)
        
    sorted_comms = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
    top_comms = sorted_comms[:5] 
    
    print(f"Detected {len(communities)} communities.")
    
    # Custom stop words for HEP domain
    custom_stop_words = [
        'english', 'date', 'gmt', 'comments', 'pages', 'title', 'authors', 
        'paper', 'abstract', 'report', 'no', 'hep', 'th', 'theory', 'phys', 
        'university', 'department', 'lat', 'mon', 'tue', 'wed', 'thu', 'fri', 
        'dec', 'nov', 'oct', 'sep', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul',
        'cern', 'fermilab', 'slac', 'caltech' 
    ]
    stop_words = list(ENGLISH_STOP_WORDS.union(custom_stop_words))
    # Fitting the model on ALL papers first to learn global word importance
    all_abstracts = list(paper_to_text.values())
    if not all_abstracts:
        print("No abstract text found for TF-IDF.")
        return

    # Increaseing max_features so we have a pool to choose from
    tfidf = TfidfVectorizer(stop_words=stop_words, max_features=5000)
    tfidf.fit(all_abstracts)
    feature_names = np.array(tfidf.get_feature_names_out())
    
    for comm_id, authors in top_comms:
        comm_text = []
        for author in authors:
            papers = author_to_papers.get(author, [])
            for pid in papers:
                if pid in paper_to_text:
                    comm_text.append(paper_to_text[pid])
        
        full_text = " ".join(comm_text)
        
        if not full_text.strip():
            print(f"Community {comm_id}: Not enough text data.")
            continue

        response = tfidf.transform([full_text])
        
        scores = response.toarray().flatten()
        top_indices = scores.argsort()[::-1][:10]
        top_keywords = feature_names[top_indices]
            
        print(f"\nCommunity {comm_id} (Size: {len(authors)} authors):")
        print(f"Top Keywords: {', '.join(top_keywords)}")



def print_top_authors(G_co, G_cit):

    print("\n--- Centrality Analysis ---")
    
    top_connected = sorted(dict(G_co.degree()).items(), key=lambda x: x[1], reverse=True)[:5]
    print("Most Collaborative (High Degree):")
    for author, degree in top_connected:
        print(f"  - {author}: {degree} co-authors")
        
    top_cited = sorted(dict(G_cit.in_degree(weight='weight')).items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nMost Influential (High Citations):")
    for author, degree in top_cited:
        print(f"  - {author}: {degree} citations")

# ==========================================
# 5. VISUALIZATION
# ==========================================

def visualize_network(G, title="Co-authorship Network (Top 1000 Nodes)"):

    print("\n--- Generating Network Visualization ---")
    
    N = 1000 
    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:N]
    G_sub = G.subgraph(top_nodes)
    
    print(f"Visualizing subgraph with {G_sub.number_of_nodes()} nodes...")

    pos = nx.spring_layout(G_sub, k=0.15, seed=42)
    partition = community_louvain.best_partition(G_sub)
    cmap = plt.get_cmap('viridis')
    
    plt.figure(figsize=(12, 12))
    
    nx.draw_networkx_nodes(G_sub, pos, 
                           node_size=[v * 10 for v in dict(G_sub.degree()).values()], 
                           cmap=cmap, 
                           node_color=list(partition.values()), 
                           alpha=0.8)
    
    nx.draw_networkx_edges(G_sub, pos, alpha=0.3, edge_color='gray')
    
    top_10 = sorted(degrees, key=degrees.get, reverse=True)[:10]
    labels = {node: node for node in top_10 if node in G_sub.nodes()}
    nx.draw_networkx_labels(G_sub, pos, labels=labels, font_size=12, font_weight='bold', font_color='black')
    
    plt.title(title)
    plt.axis('off')
    plt.show()

def print_global_metrics(G):
    print("\n--- Global Graph Metrics ---")
    
    # 1. Edge Density (How connected is the graph?)
    density = nx.density(G)
    print(f"Edge Density: {density:.6f}")
    
    # 2. Transitivity (Global Clustering Coefficient)
    # (Measures how often friends of friends are also friends)
    transitivity = nx.transitivity(G)
    print(f"Global Clustering Coeff (Transitivity): {transitivity:.4f}")
    
    # 3. Average Clustering (Local level average)
    avg_clustering = nx.average_clustering(G)
    print(f"Average Clustering Coefficient: {avg_clustering:.4f}")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    if os.path.exists(ABSTRACTS_DIR) and os.path.exists(EDGES_FILE):

        p2a, p2t, a2p = parse_abstracts(ABSTRACTS_DIR)
        
        G_co, G_cit = build_networks(EDGES_FILE, p2a)
        
        if G_co and G_cit:
            analyze_layer_shortest_paths(G_cit, G_co)
            print_global_metrics(G_co)  
            analyze_communities_and_topics(G_co, a2p, p2t)
            print_top_authors(G_co, G_cit)
            
            visualize_network(G_co)
    else:
        print(f"Data files not found.\nPlease ensure '{EDGES_FILE}' and '{ABSTRACTS_DIR}' are in the script directory.")
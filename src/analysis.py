import os
import re
import warnings
from collections import defaultdict
from itertools import combinations

import networkx as nx
import matplotlib.pyplot as plt


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

def parse_abstracts(root_dir):

    print(f"Scanning abstracts in {root_dir}...")
    
    paper_to_authors = defaultdict(list)
    paper_to_text = {}
    author_to_papers = defaultdict(list)
    
    # Common non-author words found in the dataset to filter out
    BLACKLIST = {
        'italy', 'germany', 'france', 'spain', 'russia', 'usa', 'japan', 'uk', 'england', 
        'canada', 'switzerland', 'brazil', 'india', 'china', 'korea', 'australia', 
        'mexico', 'israel', 'netherlands', 'belgium', 'sweden', 'denmark', 'finland',
        'norway', 'poland', 'austria', 'greece', 'portugal', 'hungary', 'czech republic',
        'moscow', 'rome', 'paris', 'london', 'cambridge', 'oxford', 'berlin', 'cern', 'trieste',
        'university', 'institute', 'department', 'physics', 'caltech', 'mit', 'princeton'
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
                    
                    # --- Author Parsing ---
                    auth_match = re.search(r'Authors?:\s*(.+?)(?=\n(?:Comments|Journal-ref|Subj-class|\\)|$)', content, re.DOTALL | re.IGNORECASE)

                    if auth_match:
                        # Flatten newlines and remove parentheses (affiliations)
                        raw_authors = auth_match.group(1).replace('\n', ' ')
                        raw_authors = re.sub(r'\(.*?\)', '', raw_authors)
                        
                        # Split by comma or 'and'
                        authors = re.split(r',|\sand\s', raw_authors)
                        
                        cleaned_authors = []
                        for a in authors:
                            name = a.strip()
                            if len(name) <= 1: continue
                            if name.lower() in BLACKLIST: continue
                            if "university" in name.lower() or "institute" in name.lower(): continue
                            
                            cleaned_authors.append(name)
                        
                        paper_to_authors[paper_id] = cleaned_authors
                        for auth in cleaned_authors:
                            author_to_papers[auth].append(paper_id)
                    
                    # --- Abstract Parsing ---
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
                
                for sa in source_auths:
                    for ta in target_auths:
                        if sa == ta: continue
                        if G_cit.has_edge(sa, ta):
                            G_cit[sa][ta]['weight'] += 1
                        else:
                            G_cit.add_edge(sa, ta, weight=1)
                            
    except FileNotFoundError:
        print(f"Error: Could not find {edges_file}")
        return None, None

    print(f"Co-authorship Graph: {G_co.number_of_nodes()} nodes, {G_co.number_of_edges()} edges")
    print(f"Citation Graph: {G_cit.number_of_nodes()} nodes, {G_cit.number_of_edges()} edges")
    return G_co, G_cit


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    if os.path.exists(ABSTRACTS_DIR) and os.path.exists(EDGES_FILE):

        p2a, p2t, a2p = parse_abstracts(ABSTRACTS_DIR)
        
        G_co, G_cit = build_networks(EDGES_FILE, p2a)
        

    else:
        print(f"Data files not found.\nPlease ensure '{EDGES_FILE}' and '{ABSTRACTS_DIR}' are in the script directory.")
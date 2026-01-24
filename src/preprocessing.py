import os
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

def normalize_name(name: str) -> Optional[str]:
    """Standardizes author names into 'F. Lastname' format."""
    name = name.replace(".", "").strip()
    parts = name.split()
    if len(parts) < 2:
        return None
    if len(parts) > 2 and parts[-2].islower():
        last_name = f"{parts[-2]} {parts[-1]}"
    else:
        last_name = parts[-1]
    first_initial = parts[0][0].upper()
    return f"{first_initial}. {last_name}"

def clean_text(text: str) -> str:
    """Preprocesses abstract text: removes LaTeX, math vars, and symbols."""
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"\b[a-zA-Z]+[_\d][a-zA-Z\d]*\b", " ", text)
    text = re.sub(r"\b[a-zA-Z]\b", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def parse_abstracts(root_dir: str) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, List[str]]]:
    """
    Scans a directory for .abs files to extract metadata and abstract text.
    Returns:
        paper_to_authors, paper_to_text, author_to_papers
    """
    print(f"Scanning abstracts in {root_dir}...")
    paper_to_authors = defaultdict(list)
    paper_to_text = {}
    author_to_papers = defaultdict(list)

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
                    
                # Author Extraction
                auth_match = re.search(r"Authors?:\s*(.+?)(?=\n(?:Comments|Journal-ref|Subj-class|\\)|$)", content, re.DOTALL | re.IGNORECASE)
                if auth_match:
                    raw_authors = auth_match.group(1).replace("\n", " ")
                    raw_authors = re.sub(r"\(.*?\)", "", raw_authors)
                    authors = re.split(r",|\sand\s|;", raw_authors)
                    
                    cleaned_authors = []
                    for a in authors:
                        name = a.strip()
                        if len(name) <= 2 or any(x in name.lower() for x in NON_AUTHOR_TERMS):
                            continue
                        norm = normalize_name(name)
                        if norm: cleaned_authors.append(norm)
                    
                    if cleaned_authors:
                        paper_to_authors[paper_id] = cleaned_authors
                        for auth in cleaned_authors:
                            author_to_papers[auth].append(paper_id)

                # Abstract Text Extraction
                parts = content.split("\\\\")
                abstract_candidate = parts[2] if len(parts) >= 3 else parts[-1]
                cleaned = clean_text(abstract_candidate)
                if len(cleaned) > 50:
                    paper_to_text[paper_id] = cleaned
            except Exception:
                continue

    print(f"Parsed {len(paper_to_authors)} papers.")
    return paper_to_authors, paper_to_text, author_to_papers
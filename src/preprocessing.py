import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src.constants import NON_AUTHOR_TERMS


def normalize_name(name: str) -> Optional[str]:
    """
    Standardizes author names into 'F. Lastname' format.
    Handles compound last names (e.g., 'van der Waals') by including
    all lowercase prefixes preceding the last name.
    """
    name = name.replace(".", "").strip()
    parts = name.split()
    if len(parts) < 2:
        return None
    surname_start_index = len(parts) - 1
    while surname_start_index > 1 and parts[surname_start_index - 1].islower():
        surname_start_index -= 1

    last_name = " ".join(parts[surname_start_index:])
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


def parse_abstracts(
    root_dir: str,
) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, List[str]]]:
    """
    Scans a directory for .abs files to extract metadata and abstract text.
    Returns:
        paper_to_authors, paper_to_text, author_to_papers
    """
    print(f"Scanning abstracts in {root_dir}...")
    paper_to_authors = defaultdict(list)
    paper_to_text = {}
    author_to_papers = defaultdict(list)
    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".abs"):
                continue
            paper_id = file.replace(".abs", "")
            path = os.path.join(root, file)

            try:
                with open(path, "r", encoding="latin-1") as f:
                    content = f.read()

                auth_match = re.search(
                    r"Authors?:\s*(.+?)(?=\n(?:Comments|Journal-ref|Subj-class|\\)|$)",
                    content,
                    re.DOTALL | re.IGNORECASE,
                )
                if auth_match:
                    raw_authors = auth_match.group(1).replace("\n", " ")
                    raw_authors = re.sub(r"\(.*?\)", "", raw_authors)
                    authors = re.split(r",|\sand\s|;", raw_authors)

                    cleaned_authors = []
                    for a in authors:
                        name = a.strip()
                        if len(name) <= 2:
                            continue

                        name_tokens = set(re.split(r"\W+", name.lower()))
                        if not name_tokens.isdisjoint(NON_AUTHOR_TERMS):
                            continue

                        norm = normalize_name(name)
                        if norm:
                            cleaned_authors.append(norm)

                    if cleaned_authors:
                        paper_to_authors[paper_id] = cleaned_authors
                        for auth in cleaned_authors:
                            author_to_papers[auth].append(paper_id)

                parts = content.split("\\\\")
                abstract_candidate = parts[2] if len(parts) >= 3 else parts[-1]
                cleaned = clean_text(abstract_candidate)
                if len(cleaned) > 50:
                    paper_to_text[paper_id] = cleaned
            except Exception:
                continue

    print(f"Parsed {len(paper_to_authors)} papers.")
    return paper_to_authors, paper_to_text, author_to_papers

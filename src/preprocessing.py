import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Pattern, Tuple

from joblib import Parallel, delayed
from src.constants import NON_AUTHOR_TERMS

logger = logging.getLogger(__name__)

CLEAN_LATEX_PATTERN: Pattern = re.compile(r"\\[a-zA-Z]+")
CLEAN_VARS_PATTERN: Pattern = re.compile(r"\b[a-zA-Z]+[_\d][a-zA-Z\d]*\b")
CLEAN_SINGLE_CHAR_PATTERN: Pattern = re.compile(r"\b[a-zA-Z]\b")
CLEAN_NON_ALPHA_PATTERN: Pattern = re.compile(r"[^a-zA-Z\s\-]")
CLEAN_WHITESPACE_PATTERN: Pattern = re.compile(r"\s+")

AUTH_CAPTURE_PATTERN: Pattern = re.compile(
    r"Authors?:\s*(.+?)(?=\n(?:Comments|Journal-ref|Subj-class|\\)|$)",
    re.DOTALL | re.IGNORECASE,
)
AUTH_PAREN_PATTERN: Pattern = re.compile(r"\(.*?\)")
AUTH_SPLIT_PATTERN: Pattern = re.compile(r",|\sand\s|;")
NAME_TOKEN_SPLIT_PATTERN: Pattern = re.compile(r"\W+")


def normalize_name(name: str) -> Optional[str]:
    """
    Standardizes author names into 'F. Lastname' format with safety guards.
    """
    name = name.replace(".", "").strip()
    if not name:
        return None

    parts = name.split()
    if len(parts) < 2:
        return None

    surname_start_index = len(parts) - 1
    while surname_start_index > 1 and parts[surname_start_index - 1].islower():
        surname_start_index -= 1

    last_name = " ".join(parts[surname_start_index:])

    first_initial = parts[0][0].upper() if parts[0] else ""
    if not first_initial:
        return None

    return f"{first_initial}. {last_name}"


def clean_text(text: str) -> str:
    """
    Preprocesses abstract text: removes LaTeX, math vars, and symbols.
    """
    text = CLEAN_LATEX_PATTERN.sub(" ", text)
    text = CLEAN_VARS_PATTERN.sub(" ", text)
    text = CLEAN_SINGLE_CHAR_PATTERN.sub(" ", text)
    text = CLEAN_NON_ALPHA_PATTERN.sub("", text)
    text = CLEAN_WHITESPACE_PATTERN.sub(" ", text).strip()
    return text.lower()


def _process_single_abstract(
    path: str, filename: str
) -> Optional[Tuple[str, List[str], str]]:
    """
    Helper function to process a single abstract file.
    Designed for parallel execution.
    """
    paper_id = filename.replace(".abs", "")

    try:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1", errors="replace") as f:
                content = f.read()

        authors_list = []
        auth_match = AUTH_CAPTURE_PATTERN.search(content)
        if auth_match:
            raw_authors = auth_match.group(1).replace("\n", " ")
            raw_authors = AUTH_PAREN_PATTERN.sub("", raw_authors)
            authors_split = AUTH_SPLIT_PATTERN.split(raw_authors)

            for a in authors_split:
                name = a.strip()
                if len(name) <= 2:
                    continue

                name_tokens = set(NAME_TOKEN_SPLIT_PATTERN.split(name.lower()))
                if not name_tokens.isdisjoint(NON_AUTHOR_TERMS):
                    continue

                norm = normalize_name(name)
                if norm:
                    authors_list.append(norm)

        cleaned_text = ""
        parts = [p.strip() for p in content.split("\\\\") if p.strip()]
        if len(parts) >= 2:
            abstract_candidate = parts[-1]
            cleaned = clean_text(abstract_candidate)
            if len(cleaned) > 5:
                cleaned_text = cleaned

        if not authors_list and not cleaned_text:
            return None

        return (paper_id, authors_list, cleaned_text)

    except Exception as e:
        return None


def parse_abstracts(
    root_dir: str, n_jobs: int = -1
) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, List[str]]]:
    """
    Scans a directory for .abs files to extract metadata and abstract text.

    Optimized: Uses parallel processing (joblib) to parse files concurrently.

    Args:
        root_dir (str): Directory containing .abs files.
        n_jobs (int): Number of parallel jobs (-1 for all cores).

    Returns:
        Tuple containing:
        - paper_to_authors: Dict[paper_id, List[author_names]]
        - paper_to_text: Dict[paper_id, cleaned_abstract_text]
        - author_to_papers: Dict[author_name, List[paper_ids]]
    """
    logger.info(f"Scanning abstracts in {root_dir}...")

    all_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".abs"):
                all_files.append((os.path.join(root, file), file))

    logger.info(f"Found {len(all_files)} abstract files. Parsing in parallel...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_abstract)(path, fname) for path, fname in all_files
    )

    paper_to_authors = defaultdict(list)
    paper_to_text = {}
    author_to_papers = defaultdict(list)

    count = 0
    for res in results:
        if res is None:
            continue

        pid, authors, text = res

        if authors:
            paper_to_authors[pid] = authors
            for auth in authors:
                author_to_papers[auth].append(pid)

        if text:
            paper_to_text[pid] = text

        count += 1

    logger.info(f"Parsed {count} papers successfully.")
    return paper_to_authors, paper_to_text, author_to_papers
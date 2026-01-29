import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Pattern, Tuple

from src.constants import NON_AUTHOR_TERMS

logger = logging.getLogger(__name__)

CLEAN_LATEX_PATTERN: Pattern = re.compile(r"\\[a-zA-Z]+")
CLEAN_VARS_PATTERN: Pattern = re.compile(r"\b[a-zA-Z]+[_\d][a-zA-Z\d]*\b")
CLEAN_SINGLE_CHAR_PATTERN: Pattern = re.compile(r"\b[a-zA-Z]\b")
CLEAN_NON_ALPHA_PATTERN: Pattern = re.compile(r"[^a-zA-Z\s]")
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
    Standardizes author names into 'F. Lastname' format.

    Handles compound last names (e.g., 'van der Waals') by including
    all lowercase prefixes preceding the last name.

    Args:
        name (str): Raw author name string.

    Returns:
        Optional[str]: Normalized name or None if invalid.
    """
    name = name.replace(".", "").strip()
    parts = name.split()
    if len(parts) < 2:
        return None

    # logic to handle 'van der Waals'
    surname_start_index = len(parts) - 1
    while surname_start_index > 1 and parts[surname_start_index - 1].islower():
        surname_start_index -= 1

    last_name = " ".join(parts[surname_start_index:])
    first_initial = parts[0][0].upper()

    return f"{first_initial}. {last_name}"


def clean_text(text: str) -> str:
    """
    Preprocesses abstract text: removes LaTeX, math vars, and symbols.
    Uses pre-compiled regex patterns for speed.
    """
    text = CLEAN_LATEX_PATTERN.sub(" ", text)
    text = CLEAN_VARS_PATTERN.sub(" ", text)
    text = CLEAN_SINGLE_CHAR_PATTERN.sub(" ", text)
    text = CLEAN_NON_ALPHA_PATTERN.sub("", text)
    text = CLEAN_WHITESPACE_PATTERN.sub(" ", text).strip()
    return text.lower()


def parse_abstracts(
    root_dir: str,
) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, List[str]]]:
    """
    Scans a directory for .abs files to extract metadata and abstract text.

    Args:
        root_dir (str): Directory containing .abs files.

    Returns:
        Tuple containing:
        - paper_to_authors: Dict[paper_id, List[author_names]]
        - paper_to_text: Dict[paper_id, cleaned_abstract_text]
        - author_to_papers: Dict[author_name, List[paper_ids]]
    """
    logger.info(f"Scanning abstracts in {root_dir}...")

    paper_to_authors = defaultdict(list)
    paper_to_text = {}
    author_to_papers = defaultdict(list)

    count = 0

    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".abs"):
                continue

            paper_id = file.replace(".abs", "")
            path = os.path.join(root, file)

            try:
                # 'latin-1' is common for older ArXiv datasets, but errors can happen
                with open(path, "r", encoding="latin-1") as f:
                    content = f.read()

                # Extract Authors
                auth_match = AUTH_CAPTURE_PATTERN.search(content)
                if auth_match:
                    raw_authors = auth_match.group(1).replace("\n", " ")
                    raw_authors = AUTH_PAREN_PATTERN.sub("", raw_authors)
                    authors = AUTH_SPLIT_PATTERN.split(raw_authors)

                    cleaned_authors = []
                    for a in authors:
                        name = a.strip()
                        if len(name) <= 2:
                            continue

                        # Check for non-author terms (affiliations, etc.)
                        name_tokens = set(NAME_TOKEN_SPLIT_PATTERN.split(name.lower()))
                        if not name_tokens.isdisjoint(NON_AUTHOR_TERMS):
                            continue

                        norm = normalize_name(name)
                        if norm:
                            cleaned_authors.append(norm)

                    if cleaned_authors:
                        paper_to_authors[paper_id] = cleaned_authors
                        for auth in cleaned_authors:
                            author_to_papers[auth].append(paper_id)

                # Extract Abstract Text
                # ArXiv files usually separate metadata and abstract with \\
                parts = content.split("\\\\")
                abstract_candidate = parts[2] if len(parts) >= 3 else parts[-1]

                cleaned = clean_text(abstract_candidate)
                if len(cleaned) > 50:
                    paper_to_text[paper_id] = cleaned

                count += 1

            except (IOError, UnicodeDecodeError) as e:
                logger.warning(f"Failed to read {path}: {e}")
                continue
            except Exception as e:
                logger.debug(f"Unexpected error parsing {paper_id}: {e}")
                continue

    logger.info(f"Parsed {count} papers successfully.")
    return paper_to_authors, paper_to_text, author_to_papers

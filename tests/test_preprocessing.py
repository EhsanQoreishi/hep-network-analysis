"""
Tests for preprocessing in ``src.preprocessing``: author-name normalization,
LaTeX/math noise removal from abstracts, and parallel parsing of .abs files
into paper/author/text mappings. Edge cases cover missing authors, missing
abstracts, and joblib parallelism.
"""

import os
import pytest

from src.preprocessing import clean_text, normalize_name, parse_abstracts


# =============================================================================
# ENTITY RESOLUTION TESTS (Author Names)
# =============================================================================

@pytest.mark.parametrize(
    "raw_name, expected",
    [
        # Standard names
        ("Albert Einstein", "A. Einstein"),
        ("M. K. Parikh", "M. Parikh"),
        # Complex physics surnames with lowercase particles
        ("Johannes van der Waals", "J. van der Waals"),
        ("Gerard 't Hooft", "G. 't Hooft"),
        ("L. de Broglie", "L. de Broglie"),
        # Edge cases and messy formatting
        ("   Stephen   Hawking  ", "S. Hawking"),
        ("A.B. McDonald", "A. McDonald"),
        # Invalid names (should be dropped)
        ("Plato", None),          # Single name (mononym)
        ("", None),               # Empty string
        (".", None),              # Just punctuation
    ],
)
def test_normalize_name(raw_name, expected):
    """
    Verify the author entity resolution logic.
    
    Act: Pass a raw, messy string representing an author.
    Assert: The function correctly extracts the first initial and the full surname,
    preserving lowercase particles (like 'van der') while stripping middle initials.
    """
    assert normalize_name(raw_name) == expected


# =============================================================================
# NLP TEXT CLEANING TESTS
# =============================================================================

@pytest.mark.parametrize(
    "raw_text, expected_in_output, not_expected_in_output",
    [
        # Test 1: LaTeX stripping
        (r"The value of \alpha is calculated.", "value", "alpha"),
        # Test 2: Math variable stripping (e.g., T_{\mu\nu})
        (r"We define tensor T_{\mu\nu}.", "define", "t_{\\mu\\nu}"),
        # Test 3: Hyphenated physics terms should survive; stray single chars 'x' should not.
        ("Yang-Mills theory x.", "yang-mills", " x "), 
    ],
)
def test_clean_text_noise_removal(raw_text, expected_in_output, not_expected_in_output):
    """
    Verify the regex engine correctly removes mathematical and formatting noise.
    
    Act: Clean a raw physics abstract sentence.
    Assert: The physical English words remain, but LaTeX commands and 
    math tensors are completely stripped.
    """
    cleaned = clean_text(raw_text)
    
    assert expected_in_output in cleaned
    if not_expected_in_output == "alpha":
        assert "\\alpha" not in cleaned
    else:
        assert not_expected_in_output not in cleaned


def test_clean_text_single_chars():
    """
    Verify that stray single characters (often left over from math equations) are removed.
    """
    cleaned = clean_text("Let x be a variable and y be another.")
    
    # 'x' and 'y' should be gone, 'let' and 'variable' should remain
    tokens = cleaned.split()
    assert "x" not in tokens
    assert "y" not in tokens
    assert "let" in tokens


# =============================================================================
# ABSTRACT PARSING FIXTURES & TESTS
# =============================================================================

@pytest.fixture
def sample_data(tmp_path):
    """
    Creates a temporary directory with a highly controlled dummy .abs file.
    This mimics the exact structure of an ArXiv submission, including metadata
    boundaries and LaTeX blocks.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    content = (
        "Paper: hep-th/0002031\n"
        "From: Maulik K. Parikh \n"
        "Date: Fri, 4 Feb 2000 17:04:51 GMT   (10kb)\n"
        "\n"
        "Title: Confinement and the AdS/CFT Correspondence\n"
        "Authors: D. S. Berman and Maulik K. Parikh (CERN)\n"  # Added institutional noise!
        "Comments: 12 pages, 1 figure, RevTeX\n"
        "Report-no: SPIN-1999/25, UG-1999/42\n"
        "Journal-ref: Phys.Lett. B483 (2000) 271-276\n"
        "\\\\ \n"
        "  We study the thermodynamics of the confined and unconfined phases of\n"
        "superconformal Yang-Mills in finite volume and at large N using the AdS/CFT\n"
        "correspondence. We discuss the necessary conditions for a smooth phase\n"
        "crossover and obtain an N-dependent curve for the phase boundary.\n"
        "\\\\ \n"
    )

    file_path = data_dir / "0002031.abs"
    file_path.write_text(content, encoding="latin-1")

    return data_dir


def test_parse_abstracts_authors(sample_data):
    """
    Verify the author extraction regex against complex metadata.
    
    Arrange: A dummy .abs file where authors include institutional noise '(CERN)'.
    Act: Parse the directory.
    Assert: Both authors are extracted, institutional noise is ignored, and 
    names are properly normalized.
    """
    p2a, _, a2p = parse_abstracts(str(sample_data), n_jobs=1)
    paper_id = "0002031"

    assert paper_id in p2a
    authors = p2a[paper_id]

    assert "D. Berman" in authors
    assert "M. Parikh" in authors
    assert "CERN" not in authors
    assert paper_id in a2p["M. Parikh"]


def test_parse_abstracts_text(sample_data):
    """
    Verify the abstract body is correctly isolated from the metadata header.
    
    Arrange: The dummy .abs file.
    Act: Parse the directory.
    Assert: The text block successfully captures the abstract without grabbing
    metadata lines like 'Title:' or 'Paper:'.
    """
    _, p2t, _ = parse_abstracts(str(sample_data), n_jobs=1)
    paper_id = "0002031"

    assert paper_id in p2t
    text = p2t[paper_id]

    # NLP topics must exist
    assert "thermodynamics" in text
    assert "yang-mills" in text
    
    # Metadata headers must NOT bleed into the text
    assert "paper:" not in text
    assert "title:" not in text


def test_parse_abstracts_parallel(sample_data):
    """
    Verify the Joblib parallelization produces identical results to serial execution.
    
    Arrange: Duplicate our dummy paper to create a multi-file workload.
    Act: Parse once with n_jobs=1, and once with n_jobs=2.
    Assert: The resulting data dictionaries are mathematically identical.
    """
    content = (sample_data / "0002031.abs").read_text(encoding="latin-1")
    (sample_data / "0002032.abs").write_text(content, encoding="latin-1")

    res_serial = parse_abstracts(str(sample_data), n_jobs=1)
    res_parallel = parse_abstracts(str(sample_data), n_jobs=2)

    # paper_to_authors match
    assert res_serial[0] == res_parallel[0]
    
    # paper_to_text match
    assert res_serial[1] == res_parallel[1]

    # author_to_papers match
    for auth, papers in res_serial[2].items():
        assert set(papers) == set(res_parallel[2][auth])


def test_parse_abstracts_no_authors(tmp_path):
    """
    Edge case: .abs file with no explicit 'Authors:' field but with a valid abstract body.
    
    Assert: The abstract text is captured even if no authors are extracted.
    """
    data_dir = tmp_path / "no_authors"
    data_dir.mkdir()

    content = (
        "Paper: hep-th/0000001\n"
        "Title: Sample Paper\n"
        "Comments: Just a test\n"
        "\\\\ \n"
        "  This is a simple abstract about quantum fields.\n"
        "\\\\ \n"
    )
    file_path = data_dir / "0000001.abs"
    file_path.write_text(content, encoding="utf-8")

    p2a, p2t, a2p = parse_abstracts(str(data_dir), n_jobs=1)

    assert "0000001" in p2t
    assert "quantum" in p2t["0000001"]
    # No authors or reverse mappings should be present
    assert "0000001" not in p2a
    assert a2p == {}


def test_parse_abstracts_authors_no_text(tmp_path):
    """
    Edge case: .abs file with authors metadata but no detectable abstract section.
    
    Assert: Authors are mapped correctly even if no abstract text is stored.
    """
    data_dir = tmp_path / "authors_only"
    data_dir.mkdir()

    content = (
        "Paper: hep-th/0000002\n"
        "Title: Another Paper\n"
        "Authors: Albert Einstein, Niels Bohr\n"
        "Comments: No abstract section here\n"
    )
    file_path = data_dir / "0000002.abs"
    file_path.write_text(content, encoding="utf-8")

    p2a, p2t, a2p = parse_abstracts(str(data_dir), n_jobs=1)

    pid = "0000002"
    assert pid in p2a
    assert "A. Einstein" in p2a[pid] or "Albert Einstein" in p2a[pid]
    assert pid in a2p[next(iter(a2p.keys()))]
    # There should be no abstract text recorded
    assert pid not in p2t
import os

import pytest

from src.preprocessing import clean_text, normalize_name, parse_abstracts


def test_normalize_name():
    """Test author name standardization with various edge cases."""
    assert normalize_name("Albert Einstein") == "A. Einstein"
    assert normalize_name("Gerard 't Hooft") == "G. 't Hooft"
    assert normalize_name("Johannes van der Waals") == "J. van der Waals"
    assert normalize_name("  M.   K.   Parikh  ") == "M. Parikh"
    assert normalize_name("Plato") is None
    assert normalize_name("") is None


def test_clean_text():
    """Test abstract text cleaning (LaTeX & Symbol removal)."""
    raw_latex = r"The value of \alpha is calculated using \frac{1}{2}."
    cleaned = clean_text(raw_latex)
    assert "\\alpha" not in cleaned
    assert "frac" not in cleaned
    assert "value" in cleaned
    assert "yang-mills" in clean_text("Yang-Mills theory")


@pytest.fixture
def sample_data(tmp_path):
    """
    Creates a temporary dummy .abs file with valid ArXiv structure.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    content = (
        "Paper: hep-th/0002031\n"
        "From: Maulik K. Parikh \n"
        "Date: Fri, 4 Feb 2000 17:04:51 GMT   (10kb)\n"
        "\n"
        "Title: Confinement and the AdS/CFT Correspondence\n"
        "Authors: D. S. Berman and Maulik K. Parikh\n"
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
    Test if authors are extracted correctly.
    """
    p2a, _, a2p = parse_abstracts(str(sample_data), n_jobs=1)

    paper_id = "0002031"

    assert paper_id in p2a
    authors = p2a[paper_id]

    assert "D. Berman" in authors
    assert "M. Parikh" in authors
    assert paper_id in a2p["M. Parikh"]


def test_parse_abstracts_text(sample_data):
    """Test if abstract text is extracted and cleaned."""
    _, p2t, _ = parse_abstracts(str(sample_data), n_jobs=1)

    paper_id = "0002031"
    assert paper_id in p2t

    text = p2t[paper_id]

    assert "thermodynamics" in text
    assert "yang-mills" in text
    assert "adscft" in text
    assert "Paper:" not in text
    assert "Title:" not in text


def test_parse_abstracts_parallel(sample_data):
    """
    Verify that parallel processing (Module 2 optimization)
    returns identical results to serial execution.
    """
    content = (sample_data / "0002031.abs").read_text(encoding="latin-1")
    (sample_data / "0002032.abs").write_text(content, encoding="latin-1")

    res_serial = parse_abstracts(str(sample_data), n_jobs=1)
    res_parallel = parse_abstracts(str(sample_data), n_jobs=2)

    assert res_serial[0] == res_parallel[0]
    assert res_serial[1] == res_parallel[1]

    for auth, papers in res_serial[2].items():
        assert set(papers) == set(res_parallel[2][auth])

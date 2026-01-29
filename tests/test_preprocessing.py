import pytest

from src.preprocessing import clean_text, normalize_name, parse_abstracts


def test_normalize_name():
    """
    Test author name standardization.
    Lesson 04: Use simple 'assert' statements instead of self.assertEqual.
    """
    assert normalize_name("Albert Einstein") == "A. Einstein"
    assert normalize_name("Gerard 't Hooft") == "G. 't Hooft"
    assert normalize_name("Johannes van der Waals") == "J. van der Waals"
    assert normalize_name("Plato") is None
    assert normalize_name("") is None


def test_clean_text():
    """
    Test abstract text cleaning (LaTeX removal).
    """
    raw_latex = r"The value of \alpha is calculated using \frac{1}{2}."
    cleaned = clean_text(raw_latex)
    assert "\\alpha" not in cleaned
    assert "frac" not in cleaned

    raw_math = "Let x and y be variables corresponding to e_n."
    cleaned_math = clean_text(raw_math)
    assert " x " not in cleaned_math
    assert "e_n" not in cleaned_math


@pytest.fixture
def sample_data(tmp_path):
    """
    Pytest fixture to create a temporary .abs file.
    'tmp_path' is automatically cleaned up after the test.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    content = (
        "------------------------------------------------------------------------------\n"
        "\\\\ \n"
        "Paper: hep-th/9901001\n"
        "From: Spock <spock@vulcan.edu>\n"
        "Date: Mon, 1 Jan 1999 12:00:00 GMT   (10kb)\n"
        "\n"
        "Title: Logic in High Energy Physics\n"
        "Authors: A. Einstein, R. Feynman and J. van der Waals\n"
        "Comments: 12 pages\n"
        "\\\\ \n"
        "  We discuss the logic of quantum gravity using variable e_n and alpha.\n"
        "  The results show significant improvement.\n"
        "\\\\ \n"
    )

    (data_dir / "9901001.abs").write_text(content, encoding="latin-1")

    return data_dir


def test_parse_abstracts_authors(sample_data):
    """Test if authors are correctly extracted and normalized."""
    p2a, _, a2p = parse_abstracts(str(sample_data))

    authors = p2a["9901001"]
    assert "A. Einstein" in authors
    assert "R. Feynman" in authors
    assert "J. van der Waals" in authors
    assert "9901001" in a2p["A. Einstein"]


def test_parse_abstracts_text(sample_data):
    """Test if abstract text is extracted and cleaned."""
    _, p2t, _ = parse_abstracts(str(sample_data))

    text = p2t.get("9901001", "")
    assert "discuss the logic" in text
    assert "quantum gravity" in text
    assert "e_n" not in text

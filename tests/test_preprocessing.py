import os
import shutil
import tempfile
import unittest

from src.preprocessing import clean_text, normalize_name, parse_abstracts


class TestPreprocessing(unittest.TestCase):
    def test_normalize_name(self):
        self.assertEqual(normalize_name("Albert Einstein"), "A. Einstein")
        self.assertEqual(normalize_name("Gerard 't Hooft"), "G. 't Hooft")
        self.assertEqual(normalize_name("Johannes van der Waals"), "J. van der Waals")
        self.assertIsNone(normalize_name("Plato"))
        self.assertIsNone(normalize_name(""))

    def test_clean_text(self):
        raw_latex = r"The value of \alpha is calculated using \frac{1}{2}."
        cleaned = clean_text(raw_latex)
        self.assertNotIn("\\alpha", cleaned)
        self.assertNotIn("frac", cleaned)
        raw_math = "Let x and y be variables corresponding to e_n."
        cleaned_math = clean_text(raw_math)
        self.assertNotIn(" x ", cleaned_math)
        self.assertNotIn("e_n", cleaned_math)

    def setUp(self):
        """Create a temporary directory with fake .abs files for testing."""
        self.test_dir = tempfile.mkdtemp()

        self.sample_content = (
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
        with open(os.path.join(self.test_dir, "9901001.abs"), "w") as f:
            f.write(self.sample_content)

    def tearDown(self):
        """Remove the temporary directory after tests."""
        shutil.rmtree(self.test_dir)

    def test_parse_abstracts_authors(self):
        """Test if authors are correctly extracted and normalized."""
        p2a, _, a2p = parse_abstracts(self.test_dir)
        authors = p2a["9901001"]
        self.assertIn("A. Einstein", authors)
        self.assertIn("R. Feynman", authors)
        self.assertIn("J. van der Waals", authors)
        self.assertIn("9901001", a2p["A. Einstein"])

    def test_parse_abstracts_text(self):
        """Test if abstract text is extracted and cleaned."""
        _, p2t, _ = parse_abstracts(self.test_dir)
        text = p2t.get("9901001", "")
        self.assertIn("discuss the logic", text)
        self.assertIn("quantum gravity", text)
        self.assertNotIn("e_n", text)


if __name__ == "__main__":
    unittest.main()

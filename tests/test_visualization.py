import os
import shutil
import unittest

import networkx as nx

from src.visualization import visualize_network


class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.G = nx.karate_club_graph()
        self.test_file = "tests/temp_viz/map.html"
        os.makedirs(os.path.dirname(self.test_file), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(os.path.dirname(self.test_file))

    def test_html_generation(self):
        visualize_network(self.G, title=self.test_file)
        self.assertTrue(os.path.exists(self.test_file))

        with open(self.test_file, "r") as f:
            content = f.read()
            self.assertIn("<html>", content.lower())


if __name__ == "__main__":
    unittest.main()

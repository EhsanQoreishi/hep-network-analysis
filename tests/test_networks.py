import os
import shutil
import unittest

from src.networks import build_networks


class TestNetworks(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_net_data"
        os.makedirs(self.test_dir, exist_ok=True)
        self.edges_file = os.path.join(self.test_dir, "fake_edges.txt")

        with open(self.edges_file, "w") as f:
            f.write("P1 P2\n")
            f.write("P2 P3\n")

        self.mock_p2a = {
            "P1": ["A. User"],
            "P2": ["B. User"],
            "P3": ["A. User", "C. User"],
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_build_networks_logic(self):
        G_co, G_cit = build_networks(self.edges_file, self.mock_p2a)
        self.assertTrue(G_co.has_edge("A. User", "C. User"))
        self.assertIn("A. User", G_co.nodes())
        self.assertTrue(G_cit.has_edge("A. User", "B. User"))


if __name__ == "__main__":
    unittest.main()

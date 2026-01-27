import os
import shutil
import unittest

import networkx as nx

from src.analysis import (
    analyze_communities_robust,
    analyze_layer_shortest_paths,
    analyze_power_law,
    analyze_spectral_properties,
    print_global_metrics,
)


class TestAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G = nx.karate_club_graph()
        cls.G_dir = nx.to_directed(cls.G)
        cls.out_dir = "tests/temp_results"
        os.makedirs(cls.out_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out_dir)

    def test_structural_metrics(self):
        try:
            print_global_metrics(self.G)
        except Exception as e:
            self.fail(f"print_global_metrics failed: {e}")

    def test_physics_metrics(self):
        try:
            analyze_power_law(self.G, name="TestGraph", output_dir=self.out_dir)
            analyze_spectral_properties(self.G, output_dir=self.out_dir)
        except Exception as e:
            self.fail(f"Physics analysis failed: {e}")
        self.assertTrue(
            os.path.exists(os.path.join(self.out_dir, "spectral_density_entropy.png"))
        )

    def test_communities(self):
        mock_auth_to_papers = {str(n): ["P1"] for n in self.G.nodes()}
        mock_paper_to_text = {"P1": "quantum physics theory model"}

        partition = analyze_communities_robust(
            self.G, mock_auth_to_papers, mock_paper_to_text, n_iterations=2
        )
        self.assertIsInstance(partition, dict)
        self.assertEqual(len(partition), len(self.G.nodes()))

    def test_cross_layer(self):
        avg_dist = analyze_layer_shortest_paths(
            self.G_dir, self.G, output_dir=self.out_dir
        )
        self.assertIsInstance(avg_dist, float)


if __name__ == "__main__":
    unittest.main()

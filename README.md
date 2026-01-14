# High Energy Physics (Hep-Th) Network Analysis

## Project Overview
This project performs a structural analysis of the High Energy Physics - Theory (Hep-Th) citation and co-authorship networks from the arXiv dataset. It explores the social structure of scientific collaboration and the flow of citations using Graph Theory and Natural Language Processing (NLP).

## Key Features
* **Robust Data Parsing:** Custom pipeline with regex to clean false positives (e.g., removing affiliations like "Italy") and normalize author names (e.g., merging "Edward Witten" -> "E. Witten").
* **Dual-Layer Network Construction:**
    * **Co-authorship:** Undirected, unweighted graph of collaboration.
    * **Citation:** Directed graph with **fractional weighting** (1/N authors) to prevent combinatorial bias in large collaborations.
* **Community Detection:** Uses the **Louvain algorithm** to identify social research cliques.
* **Topic Modeling:** Applies **Global TF-IDF** (trained on the full corpus) to extract mathematically distinct keywords for each community (e.g., "Chern-Simons", "Supergravity").
* **Cross-Layer Analysis:** Computes the "social distance" (shortest path in co-authorship) between authors who cite each other.

## Results (Latest Run)
* **Nodes:** ~9,782 Authors (cleaned and normalized)
* **Edges:** ~23,518 Co-authorship connections
* **Communities Detected:** 701 (distinct research sub-fields)
* **Avg Social Distance for Citations:** 3.40 steps (indicating citations closely follow social ties)
* **Top Influencer:** E. Witten (Highest weighted citation impact: ~7776)

## Visualizations

### 1. Interactive Network Visualization
Click the image below to explore the interactive graph (zoom, pan, and hover over nodes to see connections):

[![Interactive Graph Preview](results/interactive_preview.png)](https://EhsanQoreishi.github.io/hep-network-analysis/hep_network_interactive.html)
*(Note: This links to a live HTML page hosted on GitHub Pages. Requires a modern web browser.)*

### 2. Cross-Layer Social Distance
This histogram shows the shortest path length in the Co-authorship graph for every pair of authors connected in the Citation graph. The average distance is **3.40**, suggesting that influence in HEP-Th spreads through relatively tight social circles.

![Social Distance Histogram](results/path_distribution.png)
*(Figure: Distribution of shortest path lengths between citing authors. Dashed line represents the mean.)*

## Usage
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run analysis:
    ```bash
    python src/analysis.py
    ```
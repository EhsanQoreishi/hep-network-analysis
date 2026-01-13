# High Energy Physics (Hep-Th) Network Analysis

## Project Overview
This project performs a structural analysis of the High Energy Physics - Theory (Hep-Th) citation and co-authorship networks from the arXiv dataset. It explores the social structure of scientific collaboration and the flow of citations using Graph Theory and Natural Language Processing (NLP).

## Key Features
* **Data Parsing:** robust parsing of semi-structured arXiv abstract files.
* **Network Construction:** Builds both undirected Co-authorship and directed Citation graphs.
* **Community Detection:** Uses the **Louvain algorithm** to identify social research cliques.
* **Topic Modeling:** Applies **TF-IDF** to extract dominant keywords for each community (e.g., "String Theory", "Branes").
* **Layer Analysis:** Computes the "social distance" between authors who cite each other.

## Results
* **Nodes:** ~14,000 Authors
* **Communities Detected:** 1,340 (indicating highly fragmented collaborative groups).
* **Global Clustering Coefficient:** 0.50 (High social clustering).
* **Top Influencer:** Edward Witten (Highest citation count).

## Visualizations
The project generates force-directed graph visualizations to reveal the "hairball" structure of top collaborations and histograms of path lengths.

## Usage
1.  Install dependencies: `pip install -r requirements.txt`
2.  Run analysis: `python src/analysis.py`
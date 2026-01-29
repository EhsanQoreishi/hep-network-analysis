# High-Energy Physics Network Analysis (HEP-Th)

A computational physics project analyzing the structure, dynamics, and social properties of the ArXiv High-Energy Physics Theory (HEP-Th) citation network.

This project implements a reproducible pipeline to Extract, Transform, and Load (ETL) raw arXiv data, construct multi-layer networks (citation and co-authorship), and apply statistical mechanics metrics (Power Laws, Spectral Entropy) to understand scientific collaboration.

## 🕸️ Network Visualization

Below is a preview of the **Giant Component** of the co-authorship network.
*(Click the image to open the full interactive visualization if hosted)*

[![Interactive Graph Preview](output/map_preview.png)](https://EhsanQoreishi.github.io/hep-network-analysis/results/interactive_map.html)


## 🚀 Features

* **Data Parsing**: Custom ETL pipeline to parse unstructured `.abs` abstract files and edge lists.
* **Network Construction**: Builds both **Citation** (Directed) and **Co-authorship** (Undirected/Weighted) graphs.
* **Physics Analysis**:
    * **Scale-Free Dynamics**: Statistical fitting of degree distributions to Power Law ($P(k) \sim k^{-\gamma}$).
    * **Spectral Properties**: Computation of Von Neumann Entropy and Algebraic Connectivity ($\lambda_2$) using the Graph Laplacian.
    * **Robustness**: Simulation of random failures vs. targeted attacks (percolation theory).
* **Automation**: Full workflow managed by **Snakemake**.

## 📂 Project Structure

    .
    ├── data/                   # Raw datasets
    ├── logs/                   # Execution logs
    ├── results/                # Generated scientific outputs
    ├── src/                    # Source code modules
    │   ├── analysis/           # Physics & Topology logic
    │   │   ├── communities.py  # Louvain community detection
    │   │   ├── physics.py      # Power laws & Robustness
    │   │   └── structural.py   # Centrality & Path metrics
    │   ├── constants.py        # Project-wide constants
    │   ├── networks.py         # Graph construction logic
    │   ├── preprocessing.py    # ETL & Text cleaning
    │   └── visualization.py    # Plotting & PyVis generation
    ├── tests/                  # Pytest suite
    ├── environment.yml         # Conda environment definition
    ├── main.py                 # CLI entry point
    └── Snakefile               # Automated workflow pipeline

## 🛠️ Installation

This project uses **Conda** for environment management and is optimized for Apple Silicon (M1/M2/M3) and standard architectures.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/EhsanQoreishi/hep-network-analysis.git
    cd hep-network-analysis
    ```

2.  **Create the environment:**
    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the environment:**
    ```bash
    conda activate hep_network_analysis
    ```

## 📊 Usage

### Automated Pipeline (Recommended)
This project uses **Snakemake** to automate the entire analysis. It checks for file changes and only runs necessary steps.

```bash
snakemake -c1
```
* `-c1`: Uses 1 CPU core. Increase this (e.g., `-c4`) for parallel execution.

### Manual Execution (CLI)
You can also run the script manually with custom arguments:

```bash
python main.py --data data/cit-HepTh.txt --abstracts data/cit-HepTh-abstracts --output results/
```

## ✅ Testing

Strict software engineering standards are enforced using `pytest`. The suite covers data cleaning logic, network integrity, and physics calculations.

To run the tests:
```bash
pytest tests/  
```

## 🔬 Scientific Results & Interpretation

The automated pipeline generates the following physics analysis in the `results/` folder:

### 1. Scale-Free Topology (`social_layer_power_law_fit.png`)
The degree distribution $P(k)$ of the collaboration network fits a **Power Law** ($P(k) \propto k^{-\gamma}$), confirming the **Scale-Free** nature of scientific collaboration.
* **Observation:** A straight line on the log-log plot indicates that a few "hub" authors have a disproportionately large number of collaborators (preferential attachment).
* **Implication:** The network is driven by a "rich-get-richer" mechanism where established scientists attract more new connections than isolated ones.

### 2. Network Robustness & Percolation (`network_robustness.png`)
We simulate network disintegration to test resilience:
* **Random Failures:** The Giant Component size remains stable when nodes are removed randomly. This confirms the network is **robust to errors**.
* **Targeted Attacks:** The network collapses rapidly when high-degree nodes (hubs) are removed. This reveals a critical **fragility to targeted attacks**.

### 3. Spectral Properties (`spectral_density_entropy.png`)
The spectrum of the Graph Laplacian is analyzed to compute the **Von Neumann Entropy** ($S_{VN}$).
* **Algebraic Connectivity ($\lambda_2$):** The non-zero second eigenvalue indicates the network is connected (at the giant component level).
* **Entropy:** The calculated entropy value reflects the structural complexity and the presence of community clusters within the graph.

### 4. Community Structure (`interactive_map.html`)
Using the **Louvain algorithm**, we detect distinct communities (colored clusters in the map).
* These communities likely correspond to specific sub-fields within High Energy Physics (e.g., String Theory vs. Phenomenology).
* The visualization highlights the "Small-World" property: dense local clustering with short path lengths connecting distant nodes.

## 📚 Data Source

* **Citation Network**: [SNAP: ArXiv HEP-TH](https://snap.stanford.edu/data/cit-HepTh.html)
* **Abstracts**: [Cornell ArXiv Metadata](https://arxiv.org/help/bulk_data)

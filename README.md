# HEP-Th Multiplex Network Analysis

A reproducible pipeline for building and analyzing a **two-layer author multiplex** from the arXiv High-Energy Physics Theory (HEP-Th) corpus: a **social layer** (undirected weighted co-authorship) and an **intellectual layer** (directed weighted citations) on a shared aligned giant connected component of authors.

The code parses abstract files and citation edges, constructs both networks, aligns them to a common author set, restricts to the GCC, and runs single-layer topology, cross-layer coupling, null-model comparison, Louvain community detection with TF-IDF semantics, and physics-inspired diagnostics (spectral entropy, robustness, power-law fits). All results are written to `results/` and logged under `logs/`.

---

## Network visualization

The **giant component** of the co-authorship network can be explored interactively:

**[View interactive network map](https://ehsanqoreishi.github.io/hep-network-analysis/results/interactive_map.html)**

The map shows the top 500 authors by degree, colored by Louvain community. On the full HEP-Th run (see `logs/run_info.log`): average modularity **Q ≈ 0.77** (social) with stability **ARI ≈ 0.61**, average clustering **⟨C⟩ ≈ 0.437** (social) and **≈ 0.081** (citation, directed), and mean co-authorship path length **≈ 3.23** hops (weighted **≈ 1.39**).

---

## Features

- **ETL**: Parse `.abs` abstract files and citation edge list; extract author lists and abstract text; normalize author names.
- **Two layers**: Co-authorship via bipartite projection (B·Bᵀ); citation via paper→author expansion and aggregation. Edge lengths ℓ = 1/w for distance-based metrics.
- **Alignment & GCC**: Intersect author sets, restrict to the giant connected component of the aligned social layer; all analysis on this core.
- **Single-layer topology**: Density (undirected/directed), clustering, transitivity; strength–degree scaling (β); power-law degree fits; centrality (betweenness, closeness, PageRank).
- **Cross-layer**: Spearman correlation co-authorship degree vs citation in-strength; citation PageRank vs social betweenness; shortest-path analysis (citation pairs in social layer, hops and weighted).
- **Null models**: Degree-preserving random graphs (double-edge swap); Z-score of observed vs null clustering.
- **Communities**: Louvain with multi-run stability (ARI); community size distribution; TF-IDF keywords for top communities.
- **Physics-style metrics**: Normalized Laplacian spectrum, von Neumann entropy, algebraic connectivity; robustness (random vs targeted removal); configuration-model clustering comparison.
- **Automation**: Snakemake workflow; optional parallel jobs via CLI (`--jobs`); logging to `logs/run_info.log` and `logs/run_debug.log`.

---

## Project structure

Key directories and files after cloning:

```
.
├── data/                          # Citation edges (cit-HepTh.txt), abstract tree (cit-HepTh-abstracts/)
├── logs/                          # run_info.log, run_debug.log, analysis.log
├── results/                       # PDFs, CSVs, interactive_map.html
├── src/
│   ├── analysis/                  # Analysis only (no plotting)
│   │   ├── communities.py        # Louvain, ARI, TF-IDF
│   │   ├── physics.py             # Power law, spectral, robustness, null model
│   │   └── structural.py         # Centrality, multiplex correlation, paths
│   ├── constants.py               # Stop words and shared constants
│   ├── networks.py                # Bipartite projection, citation graph
│   ├── preprocessing.py           # ETL, author normalization
│   └── visualization.py           # Plotting & PyVis (data from analysis)
├── tests/                         # Pytest (structural, physics, communities, preprocessing, networks, viz, main)
├── environment.yml                # Conda environment
├── LICENSE                        # MIT
├── main.py                        # CLI orchestrator
├── pyproject.toml                 # Package and dev dependencies
└── Snakefile                      # Single rule → all results
```

---

## Installation

**Conda (recommended):**

```bash
git clone https://github.com/EhsanQoreishi/hep-network-analysis.git
cd hep-network-analysis
conda env create -f environment.yml
conda activate hep_network_analysis
```

**Optional (CLI entry point and dev tools):**

```bash
pip install -e ".[dev]"
```

---

## Usage

**Snakemake (recommended)** — builds all targets in `results/` and logs to `logs/analysis.log`:

```bash
snakemake --cores 4
```

**CLI** — same pipeline with explicit paths and job count:

```bash
python main.py --data data/cit-HepTh.txt --abstracts data/cit-HepTh-abstracts --output results --jobs -1
```

With debug logging to console:

```bash
python main.py --data data/cit-HepTh.txt --abstracts data/cit-HepTh-abstracts --output results --jobs -1 --debug
```

If installed as a package:

```bash
hep-network-analysis --data data/cit-HepTh.txt --abstracts data/cit-HepTh-abstracts --output results --jobs -1
```

---

## Testing

Tests use synthetic graphs and temporary directories; the full SNAP dataset is not required. Every public function is covered by tests, with multiple cases per function (e.g. valid data, empty data, edge cases) so that tests exercise the actual behavior of each function, not only that it runs.

**Run the full suite:**

```bash
conda activate hep_network_analysis
pytest tests/
```

**What each test file covers:**

| File | Responsibility |
|------|----------------|
| `tests/test_structural.py` | Global metrics (density, clustering, transitivity), centralities (betweenness, closeness), strength distribution, cross-layer shortest paths, multiplex and degree correlation. Uses small graphs with known analytical properties (triangles, stars, chains). |
| `tests/test_physics.py` | Power-law fitting, spectral properties, robustness (random/targeted removal), configuration-model null comparison. Uses Barabási–Albert, star, disconnected, and clustered fixtures. |
| `tests/test_communities.py` | Louvain partition, community size distribution, ARI stability, TF-IDF keywords. Uses barbell and synthetic abstract data. |
| `tests/test_preprocessing.py` | Author name normalization, LaTeX/text cleaning, `.abs` parsing (authors, text, parallel, edge cases). |
| `tests/test_networks.py` | Bipartite co-authorship projection, citation edge mapping, self-citation removal, empty/missing data. |
| `tests/test_visualization.py` | PDF/HTML generation from analysis output; empty-data tests for all plot functions; power-law fit vs fallback behavior. Does not validate plot aesthetics. |
| `tests/test_main.py` | CLI smoke test (pipeline runs without full data). |

Fixtures are small graphs or controlled inputs with known mathematical or structural properties so that tests assert actual behavior (e.g. expected density, centrality values, correlation sign) rather than only presence of keys.

---

## Data

- **Citation edges & abstracts**: [SNAP – ArXiv HEP-Th](https://snap.stanford.edu/data/cit-HepTh.html)
- Place `cit-HepTh.txt` in `data/` and the abstract tree in `data/cit-HepTh-abstracts/` (or point `--data` and `--abstracts` accordingly).

---

## Scientific results

On the aligned GCC (N = 4,658 authors; see `logs/run_info.log`):

- **Global structure**: Social layer sparse (ρ_undir ≈ 0.00127), high clustering (⟨C⟩ ≈ 0.437); citation layer denser (ρ_dir ≈ 0.0087), directed clustering ⟨C⟩ ≈ 0.081, transitivity T ≈ 0.33.
- **Heterogeneity**: Super-linear strength–degree scaling (β_co ≈ 1.13, β_cit ≈ 1.23); heavy-tailed degree distributions (power-law α in figures and logs).
- **Cross-layer**: Positive correlation degree vs in-strength (r_S ≈ 0.65); PageRank vs betweenness (r_S ≈ 0.57); citation-linked pairs socially close (mean **3.23** hops, weighted **1.39**).
- **Null model**: Clustering well above degree-preserving random (high Z-scores).
- **Communities**: High modularity (Q ≈ 0.77 social); stable across runs (ARI ≈ 0.61); TF-IDF semantics for top communities.
- **Spectral / robustness**: Laplacian spectrum, von Neumann entropy, algebraic connectivity λ₂, diffusion-time proxy τ; robustness curves under random and targeted removal.

Exact numbers are in **`logs/run_info.log`** (summary) and **`logs/run_debug.log`** (verbose). Figures are in **`results/`**.

---

## Reproducing the full analysis

1. Clone the repo and create the Conda environment (see Installation).
2. Download HEP-Th data from SNAP; put `cit-HepTh.txt` and the abstract directory in `data/` (or set `DATA_EDGES` / `DATA_ABSTRACTS` in the Snakefile, or use `--data` / `--abstracts`).
3. Run `snakemake --cores 4` (or `python main.py ...`). All figures and tables in `results/` and the report directory will be produced; logs in `logs/`.

---

## License

MIT (see `LICENSE`).

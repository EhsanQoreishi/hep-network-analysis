# Snakemake pipeline for HEP-Th Network Analysis
# Run with: snakemake -c1

# --- Configuration ---
DATA_EDGES = "data/cit-HepTh.txt"
DATA_ABSTRACTS = "data/cit-HepTh-abstracts"
RESULTS_DIR = "results"
SCRIPT = "main.py"

# --- Target Rule ---
# Defines the files we expect the pipeline to generate
rule all:
    input:
        f"{RESULTS_DIR}/interactive_map.html",
        f"{RESULTS_DIR}/social_layer_power_law_fit.png",
        f"{RESULTS_DIR}/spectral_density_entropy.png",
        f"{RESULTS_DIR}/network_robustness.png",
        f"{RESULTS_DIR}/configuration_model_comparison.png",
        f"{RESULTS_DIR}/multiplex_centrality_correlation.png"

# --- Analysis Rule ---
# Executes main.py if data or code has changed
rule run_analysis:
    input:
        script = SCRIPT,
        edges = DATA_EDGES,
        abstracts = DATA_ABSTRACTS
    output:
        f"{RESULTS_DIR}/interactive_map.html",
        f"{RESULTS_DIR}/social_layer_power_law_fit.png",
        f"{RESULTS_DIR}/spectral_density_entropy.png",
        f"{RESULTS_DIR}/network_robustness.png",
        f"{RESULTS_DIR}/configuration_model_comparison.png",
        f"{RESULTS_DIR}/multiplex_centrality_correlation.png"
    log:
        "logs/analysis.log"
    shell:
        """
        python {input.script} \
            --data {input.edges} \
            --abstracts {input.abstracts} \
            --output {RESULTS_DIR} \
            --debug > {log} 2>&1
        """
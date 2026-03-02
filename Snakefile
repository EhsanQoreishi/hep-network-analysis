# =============================================================================
# Configuration & Inputs
# =============================================================================
DATA_EDGES = "data/cit-HepTh.txt"
DATA_ABSTRACTS = "data/cit-HepTh-abstracts"
RESULTS_DIR = "results"
SCRIPT = "main.py"

# =============================================================================
# Expected Outputs (DRY Optimization)
# By defining this list once, we avoid duplicating 22 lines of code in our rules.
# =============================================================================
EXPECTED_OUTPUTS = [
    f"{RESULTS_DIR}/interactive_map.html",
    f"{RESULTS_DIR}/social_layer_degree_distribution_powerlaw.pdf",
    f"{RESULTS_DIR}/citation_layer_degree_distribution_powerlaw.pdf",
    f"{RESULTS_DIR}/degree_vs_instrength_social_vs_citation.pdf",
    f"{RESULTS_DIR}/social_layer_strength_degree_correlation.pdf",
    f"{RESULTS_DIR}/citation_layer_strength_degree_correlation.pdf",
    f"{RESULTS_DIR}/cross_layer_path_distribution_hops.pdf",
    f"{RESULTS_DIR}/cross_layer_path_distribution_weighted.pdf",
    f"{RESULTS_DIR}/multiplex_pagerank_vs_betweenness.pdf",
    f"{RESULTS_DIR}/social_layer_spectral_density_entropy.pdf",
    f"{RESULTS_DIR}/citation_layer_spectral_density_entropy.pdf",
    f"{RESULTS_DIR}/social_layer_network_robustness.pdf",
    f"{RESULTS_DIR}/citation_layer_network_robustness.pdf",
    f"{RESULTS_DIR}/social_layer_configuration_model_clustering.pdf",
    f"{RESULTS_DIR}/citation_layer_configuration_model_clustering.pdf",
    f"{RESULTS_DIR}/centrality_betweenness_closeness_social.pdf",
    f"{RESULTS_DIR}/social_layer_community_size_distribution.pdf",
    f"{RESULTS_DIR}/citation_layer_community_size_distribution.pdf",
    f"{RESULTS_DIR}/social_layer_top_communities_tfidf.csv",
    f"{RESULTS_DIR}/citation_layer_top_communities_tfidf.csv",
    f"{RESULTS_DIR}/top_betweenness_social.csv",
    f"{RESULTS_DIR}/top_closeness_social.csv"
]

# =============================================================================
# Rules
# =============================================================================

rule all:
    input:
        EXPECTED_OUTPUTS

rule run_analysis:
    input:
        script=SCRIPT,
        edges=DATA_EDGES,
        abstracts=DATA_ABSTRACTS
    output:
        EXPECTED_OUTPUTS
    log:
        "logs/analysis.log"
    threads: 4
    shell:
        """
        python {input.script} \
            --data {input.edges} \
            --abstracts {input.abstracts} \
            --output {RESULTS_DIR} \
            --jobs {threads} \
            --debug > {log} 2>&1
        """
DATA_EDGES = "data/cit-HepTh.txt"
DATA_ABSTRACTS = "data/cit-HepTh-abstracts"
RESULTS_DIR = "results"
SCRIPT = "main.py"

rule all:
    input:
        f"{RESULTS_DIR}/interactive_map.html",
        f"{RESULTS_DIR}/social_layer_power_law_fit.png",
        f"{RESULTS_DIR}/spectral_density_entropy.png",
        f"{RESULTS_DIR}/network_robustness.png"

rule run_analysis:
    input:
        script = SCRIPT,
        edges = DATA_EDGES,
        abstracts = DATA_ABSTRACTS
    output:
        f"{RESULTS_DIR}/interactive_map.html",
        f"{RESULTS_DIR}/social_layer_power_law_fit.png",
        f"{RESULTS_DIR}/spectral_density_entropy.png",
        f"{RESULTS_DIR}/network_robustness.png"
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
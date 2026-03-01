"""
Filtering sets and stop-words for the HEP-Th pipeline.
"""

from typing import Set

# When parsing raw ArXiv .abs files, the 'Authors:' field is often polluted with 
# institutional affiliations, countries, or collaboration group names. 
# We use this set during the entity resolution phase to filter out these false positives,
# ensuring our network nodes strictly represent individual human researchers.
NON_AUTHOR_TERMS: Set[str] = {
    # Countries & Regions
    "italy", "germany", "france", "spain", "russia", "usa", "japan", "uk", 
    "england", "canada", "switzerland", "brazil", "india", "china", "korea", 
    "australia", "mexico", "israel", "netherlands", "belgium", "sweden",
    
    # Major Physics Hubs & Cities
    "cern", "trieste", "moscow", "rome", "paris", "london", "berlin", "madrid",
    
    # Universities & Institutes
    "caltech", "mit", "stanford", "harvard", "princeton", "cambridge", "oxford", 
    "chicago", "columbia", "berkeley", "infn",
    
    # Academic / Departmental jargon
    "fisica", "physics", "department", "dept", "univ", "university", "institute", 
    "istituto", "nazionale", "research", "center", "centre", "lab", "laboratory", 
    "school", "college", "division", "section", "group", "collab", "collaboration", 
    
    # Common citation noise
    "et al", "et", "al", 
    
    # Foreign language academic particles
    "di", "de", "del", "dipartimento", "departamento", "complutense", "autonoma", 
    "polytechnique", "state", "tech", "technology",
}

# Standard English stop words (like 'the', 'and', 'is') are handled by scikit-learn.
# HEP abstracts overuse "physics", "theory", "model"; filtering these avoids
# TF-IDF grouping by generic language instead of sub-fields (e.g. string, branes).
CUSTOM_STOP_WORDS: Set[str] = {
    "theory",
    "model",
    "field",
    "physics",
    "results",
    "paper",
    "study",
    "analysis",
    "high",
    "energy",
    "using",
    "based",
    "shown",
}
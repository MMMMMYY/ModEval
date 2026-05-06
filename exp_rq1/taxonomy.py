"""
Unified category taxonomy for RQ1.
Populated after running section_4_1_harmonization.py.
This module is imported by both the classification and steering pipelines.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Raw input: 16 category descriptions from 4 sources
# ──────────────────────────────────────────────────────────────────────────────

RAW_CATEGORIES = [
    # Source 1: CLIP Interrogator (pharmapsychotic, 2023)
    {
        "id": "CI_Medium",
        "source": "CLIP Interrogator",
        "name": "Medium",
        "description": "artistic medium or material used",
    },
    {
        "id": "CI_Artist",
        "source": "CLIP Interrogator",
        "name": "Artist",
        "description": "named artist whose style directs the visual output",
    },
    {
        "id": "CI_Movement",
        "source": "CLIP Interrogator",
        "name": "Movement",
        "description": "artistic movement or historical style school",
    },
    {
        "id": "CI_Trending",
        "source": "CLIP Interrogator",
        "name": "Trending",
        "description": "platform or community aesthetic standard signalling contemporary visual popularity",
    },
    {
        "id": "CI_Flavor",
        "source": "CLIP Interrogator",
        "name": "Flavor",
        "description": "stylistic or thematic detail adding richness and precision",
    },
    # Source 2: Oppenlaender (2023)
    {
        "id": "OP_Style",
        "source": "Oppenlaender",
        "name": "Style modifiers",
        "description": "specifying an artistic style such as an artistic medium",
    },
    {
        "id": "OP_Quality",
        "source": "Oppenlaender",
        "name": "Quality boosters",
        "description": "indicating desired image quality or level of detail",
    },
    {
        "id": "OP_Magic",
        "source": "Oppenlaender",
        "name": "Magic terms",
        "description": "evocative or mystical term introducing creative unpredictability",
    },
    {
        "id": "OP_Repeating",
        "source": "Oppenlaender",
        "name": "Repeating terms",
        "description": "word or phrase repeated to intensify a particular quality",
    },
    # Source 3: Liu & Chilton (2022)
    {
        "id": "LC_ArtisticStyle",
        "source": "Liu & Chilton",
        "name": "Artistic style",
        "description": "named artist or art movement style keyword",
    },
    {
        "id": "LC_TechMedium",
        "source": "Liu & Chilton",
        "name": "Technical medium",
        "description": "physical or digital artistic medium and material",
    },
    {
        "id": "LC_AestheticDesc",
        "source": "Liu & Chilton",
        "name": "Aesthetic descriptors",
        "description": "atmospheric or mood-related keyword evoking emotional quality",
    },
    {
        "id": "LC_QualityTerms",
        "source": "Liu & Chilton",
        "name": "Quality terms",
        "description": "technical keyword controlling detail, resolution, or rendering quality",
    },
    # Source 4: Hao et al. (2023)
    {
        "id": "HA_Stylistic",
        "source": "Hao et al.",
        "name": "Stylistic descriptors",
        "description": "style including artist references and style names",
    },
    {
        "id": "HA_Quality",
        "source": "Hao et al.",
        "name": "Quality enhancers",
        "description": "quality improving technical fidelity and aesthetic appeal",
    },
    {
        "id": "HA_Thematic",
        "source": "Hao et al.",
        "name": "Thematic/mood terms",
        "description": "thematic or atmosphere establishing emotional tone and narrative mood",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Unified taxonomy (filled in after running section_4_1_harmonization.py)
# Keys are unified category names; values contain definition and member IDs.
# ──────────────────────────────────────────────────────────────────────────────

# This dict is populated by harmonization.py and saved to unified_taxonomy.json.
# Import that JSON for downstream use rather than hard-coding here.
UNIFIED_TAXONOMY: dict = {}

# Modifier Classification Prompt
# For: Llama 3.1 8B Instruct
# Task: Identify and classify modifier spans in T2I prompts
# Following: GoLLIE guideline-following paradigm (Sainz et al., ICLR 2024)

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert annotator for text-to-image (T2I) prompts.
Your task is to identify all modifier spans in a given prompt and classify each
one into exactly one of the eight categories defined below.

## Definition

A T2I prompt typically has two components:
- **Subject**: the core content or entity to be depicted (e.g., "a castle",
  "portrait of a woman", "a dragon").
- **Modifiers**: all other tokens that adjust the artistic style, quality,
  atmosphere, or technical properties of the generated image.

You must identify every modifier span and assign it to one of the eight
categories. Do NOT include subject tokens in your output.

**IMPORTANT — scan the entire prompt**: Real-world T2I prompts almost always
contain modifiers. Even when a prompt begins with a long subject description,
keep scanning every token to the end. The following terms are **always**
modifiers regardless of where they appear or what surrounds them:

- Platform names: `artstation`, `deviantart`, `pixiv`, `behance`, `cgsociety`
- Artist patterns: `art by`, `by [name]`, `in the style of`, standalone known
  artist names (artgerm, greg rutkowski, wlop, alphonse mucha, thomas kinkade,
  loish, norman rockwell, stanley lau, makoto shinkai, ross tran, etc.)
- Quality anchors: `highly detailed`, `intricate`, `sharp focus`, `concept art`,
  `smooth`, `8k`, `4k`, `hdr`, `hyperrealistic`, `ultra realistic`, `octane render`,
  `unreal engine`, `ray tracing`, `global illumination`
- Medium anchors: `digital painting`, `oil painting`, `illustration`, `matte painting`,
  `watercolor`, `pencil`, `photography`
- Movement anchors: `fantasy`, `cyberpunk`, `sci-fi`, `d & d`, `dnd`, `rpg`,
  `anime`, `art nouveau`, `impressionism`, `baroque`, `vaporwave`

Only return an empty list if the prompt contains **none** of the above and
consists purely of a subject description.

## Category Guidelines

**1. Artist**
Named artist references, artist-style designations, or style keywords that
direct the image toward a specific artist's creative signature.
- Include: "by [name]", "in the style of [name]", standalone artist names
- Examples: `by Greg Rutkowski`, `Thomas Kinkade`, `artgerm`,
  `in the style of Alphonse Mucha`, `painted by James Gurney`

**2. Medium**
The physical or digital artistic medium, material, or production technique
that determines the surface texture and material appearance of the image.
- Include: traditional materials, digital tools, photographic techniques
- Examples: `watercolor`, `oil painting`, `photography`, `charcoal sketch`,
  `digital painting`, `35mm film`, `pencil drawing`, `linocut`, `gouache`

**3. Movement**
An artistic movement, historical period, or collective style school that
provides a broader aesthetic framework beyond a single named artist.
- Include: named art movements, historical periods, genre labels
- Examples: `impressionism`, `baroque`, `art nouveau`, `cyberpunk`,
  `surrealism`, `street art`, `romanticism`, `vaporwave`, `dark fantasy`

**4. Trending**
Platform names or community tags used as quality and aesthetic references,
signalling conformity to the visual standards of a specific online community.
- Include: platform names, community-associated quality signals
- Examples: `artstation`, `trending on artstation`, `pixiv`,
  `deviantart`, `behance`, `instagram`, `cgsociety`, `unreal engine render`

**5. Quality**
Technical and stylistic terms specifying the desired level of detail,
rendering fidelity, or visual richness of the output, including both
quality-signalling phrases and precision-adding stylistic details.
- Include: resolution terms, detail descriptors, rendering engines,
  quality boosters, precision phrases
- Examples: `highly detailed`, `8k`, `sharp focus`, `intricate`,
  `ultra-realistic`, `octane render`, `smooth`, `masterpiece`,
  `concept art`, `high resolution`, `best quality`, `4k uhd`

**6. Atmosphere**
Atmospheric, mood-related, and thematic keywords that evoke an emotional
tone, contextual setting, or narrative quality, without referencing a
specific artist or medium.
- Include: lighting descriptors, emotional tone words, setting atmosphere,
  thematic context
- Examples: `dramatic lighting`, `ethereal`, `cinematic`, `dark and moody`,
  `golden hour`, `mysterious`, `soft ambient lighting`, `volumetric lighting`,
  `god rays`, `rim light`, `dystopian`, `epic`, `serene`

**7. Repeating**
A word or phrase repeated multiple times within the prompt to intensify
or emphasize a particular visual quality.
- Include: any intentional repetition of the same token or short phrase
- Examples: `very very very detailed`, `extremely extremely beautiful`,
  `ultra ultra realistic`

**8. Magic**
Evocative, mystical, or semantically opaque terms that introduce creative
unpredictability, producing results that are difficult to predict from the
literal meaning of the term alone.
- Include: abstract evocative terms, mystical phrases, terms with no clear
  technical meaning but a known stylistic effect in the SD community
- Examples: `control the soul`, `feel the sound`, `bokeh`, `intricate filigree`

## Disambiguation Rules

- If a term could be Artist **or** Movement (e.g., "impressionism" can be a
  movement but "by Monet" is an artist), prefer **Movement** for style-school
  names and **Artist** for named individuals.
- If a term could be Quality **or** Atmosphere (e.g., "cinematic"), prefer
  **Atmosphere** if it evokes mood/tone; prefer **Quality** if it signals
  technical fidelity (e.g., "cinematic quality" → Quality).
- If a term could be Trending **or** Quality (e.g., "artstation"), prefer
  **Trending** for platform/community names, **Quality** for generic
  quality phrases.
- **Artist names without "by"**: standalone names like `artgerm`, `wlop`,
  `greg rutkowski`, `loish`, `alphonse mucha`, `thomas kinkade` are **Artist**
  even without an explicit "by" prefix.
- **"concept art"**: classify as **Quality** (it signals rendering style and
  fidelity), not Movement.
- **"d & d" / "dnd" / "rpg" / "lotr"**: classify as **Movement** (they denote
  a thematic/genre aesthetic), not Trending.
- **"matte painting"**: classify as **Medium**, not Quality.
- **"full body", "half body", "wide angle", "close-up"**: classify as
  **Quality** (framing and composition descriptors).
- Assign each span to exactly **one** category. Do not split a multi-word
  modifier phrase into separate entries unless the words belong to clearly
  different categories.

## Few-shot Examples

**Example 1** (long prompt with many modifiers after a detailed subject):

Prompt: "portrait of a young ruggedly handsome man holding a corgi dog, soft hair, muscular, d & d, fantasy, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski"

Output:
{"modifiers": [
  {"span": "d & d", "category": "Movement", "start": 60, "end": 65},
  {"span": "fantasy", "category": "Movement", "start": 67, "end": 74},
  {"span": "intricate", "category": "Quality", "start": 76, "end": 85},
  {"span": "elegant", "category": "Atmosphere", "start": 87, "end": 94},
  {"span": "highly detailed", "category": "Quality", "start": 96, "end": 111},
  {"span": "digital painting", "category": "Medium", "start": 113, "end": 129},
  {"span": "artstation", "category": "Trending", "start": 131, "end": 141},
  {"span": "concept art", "category": "Quality", "start": 143, "end": 154},
  {"span": "smooth", "category": "Quality", "start": 156, "end": 162},
  {"span": "sharp focus", "category": "Quality", "start": 164, "end": 175},
  {"span": "illustration", "category": "Medium", "start": 177, "end": 189},
  {"span": "art by artgerm and greg rutkowski", "category": "Artist", "start": 191, "end": 224}
]}

**Example 2** (prompt with platform names and render engine):

Prompt: "fantasy paladin, dnd character portrait, full body, rpg, artstation, deviantart, global illumination ray tracing hdr render in unreal engine 5"

Output:
{"modifiers": [
  {"span": "fantasy", "category": "Movement", "start": 0, "end": 7},
  {"span": "dnd character portrait", "category": "Movement", "start": 16, "end": 38},
  {"span": "full body", "category": "Quality", "start": 40, "end": 49},
  {"span": "rpg", "category": "Movement", "start": 51, "end": 54},
  {"span": "artstation", "category": "Trending", "start": 56, "end": 66},
  {"span": "deviantart", "category": "Trending", "start": 68, "end": 78},
  {"span": "global illumination ray tracing hdr render in unreal engine 5", "category": "Quality", "start": 80, "end": 141}
]}

## Output Format

Return a JSON object with a single key "modifiers" containing a list of
objects. Each object must have exactly three fields:
- "span": the exact modifier text as it appears in the prompt (string)
- "category": one of the eight category names (string)
- "start": character index of the first character of the span (integer)
- "end": character index immediately after the last character (integer)

If the prompt contains no modifiers (subject only), return:
{"modifiers": []}

Do not include any explanation, preamble, or text outside the JSON object.
Your entire response must be valid JSON."""


# ──────────────────────────────────────────────────────────────────────────────
# USER PROMPT TEMPLATE
# ──────────────────────────────────────────────────────────────────────────────

USER_PROMPT_TEMPLATE = """Classify all modifier spans in the following T2I prompt.

Prompt: {prompt}"""


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE CODE
# ──────────────────────────────────────────────────────────────────────────────

import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load once, reuse for all prompts
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",          # single A6000
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,         # sufficient for most prompts
    do_sample=False,            # greedy decoding for consistency
    temperature=None,
    top_p=None,
    pad_token_id=tokenizer.eos_token_id,
)

VALID_CATEGORIES = {
    "Artist", "Medium", "Movement", "Trending",
    "Quality", "Atmosphere", "Repeating", "Magic"
}


def classify_modifiers(prompt: str) -> list[dict]:
    """
    Run the Llama classifier on a single prompt.
    Returns a list of modifier dicts, or [] on failure.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(prompt=prompt)},
    ]

    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    output = pipe(input_text)[0]["generated_text"]

    # Extract only the newly generated portion
    response = output[len(input_text):].strip()

    return parse_response(response, prompt)


def parse_response(response: str, prompt: str) -> list[dict]:
    """
    Parse the model's JSON output.
    Falls back to empty list on any parse error.
    """
    # Strip markdown code fences if model added them
    response = re.sub(r"^```(?:json)?\s*", "", response)
    response = re.sub(r"\s*```$", "", response)
    response = response.strip()

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON object substring
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    modifiers = data.get("modifiers", [])
    if not isinstance(modifiers, list):
        return []

    # Validate and clean each modifier entry
    cleaned = []
    for m in modifiers:
        if not isinstance(m, dict):
            continue
        span     = m.get("span", "")
        category = m.get("category", "")
        start    = m.get("start")
        end      = m.get("end")

        # Basic validation
        if not span or category not in VALID_CATEGORIES:
            continue
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start < 0 or end > len(prompt) or start >= end:
            continue
        # Verify span matches prompt at stated position
        if prompt[start:end] != span:
            # Try soft match: find the span in the prompt
            idx = prompt.find(span)
            if idx == -1:
                continue
            start, end = idx, idx + len(span)

        cleaned.append({
            "span": span,
            "category": category,
            "start": start,
            "end": end,
        })

    return cleaned


# ──────────────────────────────────────────────────────────────────────────────
# BATCH INFERENCE
# ──────────────────────────────────────────────────────────────────────────────

def classify_batch(prompts: list[str],
                   batch_size: int = 8) -> list[list[dict]]:
    """
    Classify a list of prompts. Returns a list of modifier lists.
    Uses simple sequential batching; increase batch_size if VRAM allows.
    """
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        for prompt in batch:
            results.append(classify_modifiers(prompt))
        if i % 1000 == 0:
            print(f"Processed {i}/{len(prompts)}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# QUICK TEST — run before full 80K inference
# ──────────────────────────────────────────────────────────────────────────────

TEST_CASES = [
    # Case 1: rich modifier prompt
    (
        "a beautiful woman in a fantasy garden, digital painting, "
        "concept art, by Greg Rutkowski and Alphonse Mucha, "
        "highly detailed, sharp focus, artstation, dramatic lighting, "
        "cinematic, 8k",
        {
            "Artist":     ["by Greg Rutkowski and Alphonse Mucha"],
            "Medium":     ["digital painting"],
            "Movement":   [],
            "Trending":   ["artstation"],
            "Quality":    ["concept art", "highly detailed", "sharp focus", "8k"],
            "Atmosphere": ["dramatic lighting", "cinematic"],
            "Repeating":  [],
            "Magic":      [],
        }
    ),
    # Case 2: medium + movement dominant
    (
        "a mountain landscape, watercolor, impressionism, "
        "very very detailed, golden hour, trending on artstation",
        {
            "Artist":     [],
            "Medium":     ["watercolor"],
            "Movement":   ["impressionism"],
            "Trending":   ["trending on artstation"],
            "Quality":    [],
            "Atmosphere": ["golden hour"],
            "Repeating":  ["very very detailed"],
            "Magic":      [],
        }
    ),
    # Case 3: subject only
    (
        "a red apple on a table",
        {"modifiers": []}
    ),
    # Case 4: magic terms
    (
        "a portrait, oil painting, control the soul, "
        "highly detailed, by Thomas Kinkade",
        {
            "Artist":     ["by Thomas Kinkade"],
            "Medium":     ["oil painting"],
            "Quality":    ["highly detailed"],
            "Magic":      ["control the soul"],
        }
    ),
]

if __name__ == "__main__":
    print("=== Running test cases ===\n")
    for prompt, expected in TEST_CASES:
        result = classify_modifiers(prompt)
        print(f"Prompt: {prompt[:60]}...")
        print(f"Result: {json.dumps(result, indent=2)}")
        print()

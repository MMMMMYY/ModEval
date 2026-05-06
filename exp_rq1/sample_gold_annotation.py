"""
sample_gold_annotation.py
=========================
Stratified sampling of 150 prompts for gold annotation.

Strategy:
  - Rare categories (Magic, Repeating): oversample to ensure coverage
  - Common categories: sample proportionally
  - Empty outputs: include ~15 to validate false-negative rate
  - Complex prompts (4+ categories): include ~15 for hard cases
  - Dedup by prompt_id throughout

Output: outputs/4_2/gold_annotation_150.jsonl  (for annotators)
        outputs/4_2/gold_annotation_150.txt     (human-readable)
"""

import json
import random
from collections import defaultdict
from pathlib import Path

JSONL_PATH  = "outputs/4_2/classified_modifiers.jsonl"
OUT_DIR     = Path("outputs/4_2")
SEED        = 42

# How many per stratum (will dedup across strata)
STRATA = {
    "Magic":     20,   # rare — oversample
    "Repeating": 20,   # rare — oversample
    "Movement":  15,
    "Medium":    15,
    "Artist":    12,
    "Trending":  12,
    "Atmosphere":12,
    "Quality":   10,
    "empty":     15,   # modifiers == []
    "complex":   15,   # prompts with 4+ distinct categories
}
TARGET = 150

random.seed(SEED)


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def get_categories(rec: dict) -> set[str]:
    return {m["category"] for m in rec.get("modifiers", [])}


def main():
    print("Loading records …")
    records = load_records(JSONL_PATH)
    print(f"Total records: {len(records):,}")

    # Build per-stratum pools
    pools: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        cats = get_categories(rec)
        if not cats:
            pools["empty"].append(rec)
        else:
            if len(cats) >= 4:
                pools["complex"].append(rec)
            for cat in cats:
                pools[cat].append(rec)

    # Sample each stratum
    selected_ids: set[int] = set()
    selected: list[dict] = []

    for stratum, n in STRATA.items():
        pool = [r for r in pools[stratum] if r["prompt_id"] not in selected_ids]
        random.shuffle(pool)
        chosen = pool[:n]
        for rec in chosen:
            selected_ids.add(rec["prompt_id"])
            selected.append(rec)
        print(f"  {stratum:<12}: sampled {len(chosen):>2} / requested {n}  "
              f"(pool size {len(pools[stratum]):,})")

    # Top up to TARGET if short
    if len(selected) < TARGET:
        remaining = [r for r in records if r["prompt_id"] not in selected_ids]
        random.shuffle(remaining)
        topup = remaining[:TARGET - len(selected)]
        for rec in topup:
            selected_ids.add(rec["prompt_id"])
            selected.append(rec)
        print(f"  top-up      : added {len(topup)} to reach {TARGET}")

    selected = selected[:TARGET]
    selected.sort(key=lambda r: r["prompt_id"])
    print(f"\nFinal sample: {len(selected)} prompts")

    # ── Write JSONL for code use ──────────────────────────────────────────────
    jsonl_out = OUT_DIR / "gold_annotation_150.jsonl"
    with open(jsonl_out, "w") as f:
        for rec in selected:
            # Strip existing modifiers — annotators fill these in
            f.write(json.dumps({
                "prompt_id": rec["prompt_id"],
                "prompt":    rec["prompt"],
                "modifiers": []          # to be filled by annotator
            }) + "\n")
    print(f"JSONL (blank) → {jsonl_out}")

    # ── Write human-readable TXT ──────────────────────────────────────────────
    txt_out = OUT_DIR / "gold_annotation_150.txt"
    CATS = ["Artist", "Medium", "Movement", "Trending",
            "Quality", "Atmosphere", "Repeating", "Magic"]

    with open(txt_out, "w") as f:
        f.write("GOLD ANNOTATION — 150 prompts\n")
        f.write("=" * 70 + "\n")
        f.write("Instructions:\n")
        f.write("  For each prompt, identify all modifier spans and their category.\n")
        f.write(f"  Categories: {', '.join(CATS)}\n")
        f.write("  Mark only modifiers, not subject descriptions.\n")
        f.write("=" * 70 + "\n\n")

        for i, rec in enumerate(selected, 1):
            f.write(f"── [{i:03d}] (id={rec['prompt_id']}) "
                    + "─" * 60 + "\n")
            f.write(f"PROMPT : {rec['prompt']}\n")
            f.write("\nYOUR ANNOTATION:\n")
            f.write("  " + "_" * 60 + "\n\n")

        # Summary
        f.write("=" * 70 + "\n")
        f.write(f"Total: {len(selected)} prompts\n")
        cat_counts = defaultdict(int)
        empty_n = 0
        for rec in selected:
            cats = get_categories(rec)
            if not cats:
                empty_n += 1
            for c in cats:
                cat_counts[c] += 1
        f.write(f"Empty (no modifiers): {empty_n}\n")
        for cat in CATS:
            f.write(f"  {cat:<12}: {cat_counts[cat]} prompts\n")

    print(f"TXT  (annotators) → {txt_out}")

    # ── Print preview ─────────────────────────────────────────────────────────
    print("\n── Preview (first 5) " + "─" * 50)
    for rec in selected[:5]:
        cats = get_categories(rec)
        print(f"  [{rec['prompt_id']:>5}] [{', '.join(sorted(cats)) or 'EMPTY'}]")
        print(f"         {rec['prompt'][:80]}{'…' if len(rec['prompt'])>80 else ''}")


if __name__ == "__main__":
    main()

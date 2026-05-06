"""
Section 4.2 — LLM-based Modifier Classification
================================================
Classifies modifiers in the 80K Stable-Diffusion-Prompts dataset
using Llama 3.1 8B Instruct with a structured three-part prompt,
then validates on a stratified 150-prompt gold-standard annotation.

Outputs
-------
outputs/4_2/
  classified_modifiers.jsonl   – one JSON record per prompt
  validation_results.json      – Kappa + per-class P/R/F1
  table_kappa_f1.csv           – Table for the paper
  confusion_matrix.pdf         – Appendix B figure

Usage
-----
# Step 1: run classification on the full dataset
python section_4_2_classification.py --mode classify

# Step 2: evaluate against gold annotations (JSON file produced by authors)
python section_4_2_classification.py --mode validate \
    --gold_path data/gold_annotations_150.json
"""

import argparse
import ast
import json
import os
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
TAXONOMY_PATH = "outputs/unified_taxonomy.json"
OUTPUT_DIR    = "outputs/4_2"
LLM_MODEL     = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 512
KAPPA_THRESHOLD = 0.75  # Landis & Koch (1977) "substantial agreement"
DATASET_NAME  = "Gustavosta/Stable-Diffusion-Prompts"
FLUSH_EVERY   = 100     # flush output file every N records

# Read HF token from cache
_token_path = os.path.expanduser("~/.cache/huggingface/token")
HF_TOKEN = open(_token_path).read().strip() if os.path.exists(_token_path) else os.environ.get("HF_TOKEN")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load validated prompt from classifier_prompt.py ──────────────────────────
# Use the pilot-tested SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, and parse_response
# rather than the simplified placeholders below.
with open(os.path.join(os.path.dirname(__file__), "classifier_prompt.py")) as _f:
    _cp_src = _f.read()

exec(compile(_cp_src.split("MODEL_ID")[0], "classifier_prompt.py", "exec"), globals())

_extra_nodes = []
for _node in ast.walk(ast.parse(_cp_src)):
    if isinstance(_node, ast.FunctionDef) and _node.name == "parse_response":
        _extra_nodes.append(_node)
    elif isinstance(_node, ast.Assign):
        for _t in _node.targets:
            if isinstance(_t, ast.Name) and _t.id == "VALID_CATEGORIES":
                _extra_nodes.append(_node)
if _extra_nodes:
    exec(compile(ast.Module(body=_extra_nodes, type_ignores=[]),
                 "classifier_prompt.py", "exec"), globals())


# ── Taxonomy helpers ──────────────────────────────────────────────────────────

def load_taxonomy(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_guidelines(taxonomy: dict, top_examples: dict[str, list[str]]) -> str:
    """
    Build the per-category guideline block for the classification prompt.
    top_examples: {unified_name: [ex1, ex2, ex3]}
    """
    lines = []
    for name, info in taxonomy.items():
        # Derive a one-sentence definition from the first member description
        defn = info["members"][0]["description"] if info["members"] else ""
        examples = top_examples.get(name, [])
        ex_str = ", ".join(f'"{e}"' for e in examples[:3]) if examples else "N/A"
        lines.append(
            f'- **{name}**: {defn}\n'
            f'  Examples: {ex_str}'
        )
    return "\n".join(lines)


# SYSTEM_PROMPT and USER_PROMPT_TEMPLATE are loaded from classifier_prompt.py above.


# ── LLM inference ─────────────────────────────────────────────────────────────

def load_model(model_id: str):
    print(f"[4.2] Loading {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        token=HF_TOKEN,
    )
    model.eval()
    return tokenizer, model


def classify_prompt(prompt_text: str, tokenizer, model) -> dict:
    """
    Run the LLM on a single prompt using the validated prompt from
    classifier_prompt.py. Returns {"modifiers": []} on failure.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(prompt=prompt_text)},
    ]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    # apply_chat_template returns a plain tensor in newer transformers
    # but a BatchEncoding in some older versions
    if hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids.to(model.device)
    else:
        input_ids = encoded.to(model.device)

    for attempt in range(3):
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(
            output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True
        ).strip()
        modifiers = parse_response(raw, prompt_text)
        if modifiers is not None:
            return {"modifiers": modifiers}
        time.sleep(0.5)

    return {"modifiers": []}


# ── Frequency extraction (for guideline examples) ─────────────────────────────

def extract_top_examples(
    classified_path: str,
    taxonomy: dict,
    top_k: int = 3,
) -> dict[str, list[str]]:
    """
    After a first-pass classification (or on a sample), extract the top-k
    most frequent modifier terms per unified category.
    Used to enrich the taxonomy guidelines with concrete examples.
    """
    from collections import Counter
    counters: dict[str, Counter] = {name: Counter() for name in taxonomy}
    with open(classified_path) as f:
        for line in f:
            record = json.loads(line)
            for m in record.get("modifiers", []):
                cat = m.get("category")
                if cat in counters:
                    counters[cat][m["span"].lower().strip()] += 1
    return {name: [t for t, _ in ctr.most_common(top_k)] for name, ctr in counters.items()}


# ── Classification pipeline ───────────────────────────────────────────────────

def run_classification(taxonomy: dict, resume: bool = True) -> None:
    category_names = list(taxonomy.keys())

    # Placeholder examples until a first pass populates them
    placeholder_examples: dict[str, list[str]] = {name: [] for name in category_names}

    out_path = Path(OUTPUT_DIR) / "classified_modifiers.jsonl"

    # Find already-processed prompt indices for resumption
    done_ids: set[int] = set()
    if resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                rec = json.loads(line)
                done_ids.add(rec["prompt_id"])
        print(f"[4.2] Resuming — {len(done_ids)} prompts already classified.")

    print("[4.2] Loading dataset …")
    ds = load_dataset(DATASET_NAME, split="train")

    tokenizer, model = load_model(LLM_MODEL)

    with open(out_path, "a") as fout:
        for idx, example in enumerate(tqdm(ds, desc="Classifying")):
            if idx in done_ids:
                continue
            prompt_text = example["Prompt"]
            result = classify_prompt(prompt_text, tokenizer, model)
            record = {
                "prompt_id": idx,
                "prompt": prompt_text,
                "modifiers": result["modifiers"],
            }
            fout.write(json.dumps(record) + "\n")
            if idx % FLUSH_EVERY == 0:
                fout.flush()

    print(f"[4.2] Classification complete → {out_path}")


# ── Validation pipeline ───────────────────────────────────────────────────────

def run_validation(gold_path: str, taxonomy: dict) -> None:
    """
    gold_path: JSON with structure:
      [{"prompt_id": int, "prompt": str, "modifiers": [{"span":..., "category":...}]}]
    Matches by prompt_id against classified_modifiers.jsonl.
    """
    category_names = list(taxonomy.keys())
    classified_path = Path(OUTPUT_DIR) / "classified_modifiers.jsonl"

    # Load LLM classifications
    llm_records: dict[int, dict] = {}
    with open(classified_path) as f:
        for line in f:
            rec = json.loads(line)
            llm_records[rec["prompt_id"]] = rec

    # Load gold annotations
    with open(gold_path) as f:
        gold_records: list[dict] = json.load(f)

    # Align: for each span in gold, find matching span in LLM output
    y_true, y_pred = [], []

    for gold_rec in gold_records:
        pid = gold_rec["prompt_id"]
        if pid not in llm_records:
            print(f"[4.2] Warning: prompt_id {pid} not found in LLM output; skipping.")
            continue
        llm_mods = {m["span"].strip().lower(): m["category"]
                    for m in llm_records[pid].get("modifiers", [])}

        for gold_m in gold_rec.get("modifiers", []):
            span_key = gold_m["span"].strip().lower()
            true_cat = gold_m["category"]
            pred_cat = llm_mods.get(span_key, "__MISSING__")
            y_true.append(true_cat)
            y_pred.append(pred_cat if pred_cat in category_names else "__MISSING__")

    # Cohen's Kappa (filter out __MISSING__ pairs from kappa for span-level eval)
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p != "__MISSING__"]
    if valid_pairs:
        kappa = cohen_kappa_score([t for t, _ in valid_pairs],
                                  [p for _, p in valid_pairs])
    else:
        kappa = 0.0

    print(f"\n[4.2] Cohen's Kappa (span-level, exact match): {kappa:.4f}")
    if kappa < KAPPA_THRESHOLD:
        print(f"[4.2] ⚠ Kappa {kappa:.3f} < {KAPPA_THRESHOLD}. "
              "Revise taxonomy guidelines before proceeding to full classification.")
    else:
        print(f"[4.2] ✓ Kappa {kappa:.3f} ≥ {KAPPA_THRESHOLD}. "
              "Classification quality sufficient for frequency analysis.")

    # Per-category metrics
    labels_for_report = [c for c in category_names if c in y_true]
    report = classification_report(
        y_true, y_pred,
        labels=labels_for_report,
        output_dict=True,
        zero_division=0,
    )
    df_report = pd.DataFrame(report).T.round(3)
    df_report.to_csv(Path(OUTPUT_DIR) / "table_kappa_f1.csv")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels_for_report + ["__MISSING__"])
    _plot_confusion_matrix(cm, labels_for_report + ["__MISSING__"])

    # Save full validation results
    results = {
        "kappa": round(kappa, 4),
        "n_gold_spans": len(y_true),
        "n_matched_spans": len(valid_pairs),
        "per_category": {
            k: {"precision": v.get("precision"), "recall": v.get("recall"), "f1": v.get("f1-score")}
            for k, v in report.items()
            if k in category_names
        },
    }
    with open(Path(OUTPUT_DIR) / "validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"[4.2] Validation results saved → {OUTPUT_DIR}/")
    print("\nPer-category F1:")
    for cat, vals in results["per_category"].items():
        print(f"  {cat:30s}  P={vals['precision']:.3f}  R={vals['recall']:.3f}  F1={vals['f1']:.3f}")


def _plot_confusion_matrix(cm: np.ndarray, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title("LLM vs. Gold-standard Confusion Matrix (Section 4.2)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    path = Path(OUTPUT_DIR) / "confusion_matrix.pdf"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[4.2] Confusion matrix saved → {path}")


# ── Frequency statistics (used in 4.3 Step 1) ────────────────────────────────

def compute_frequency_stats(taxonomy: dict) -> pd.DataFrame:
    """
    Compute f_p(c) and f_m(c) from classified_modifiers.jsonl.
    Saves outputs/4_2/frequency_stats.csv for use by section_4_3.
    """
    classified_path = Path(OUTPUT_DIR) / "classified_modifiers.jsonl"
    category_names = list(taxonomy.keys())

    n_prompts_total = 0
    prompt_counts: dict[str, int] = {c: 0 for c in category_names}
    modifier_counts: dict[str, int] = {c: 0 for c in category_names}
    n_modifiers_total = 0

    with open(classified_path) as f:
        for line in f:
            rec = json.loads(line)
            n_prompts_total += 1
            seen_cats: set[str] = set()
            for m in rec.get("modifiers", []):
                cat = m.get("category")
                if cat in category_names:
                    modifier_counts[cat] += 1
                    n_modifiers_total += 1
                    seen_cats.add(cat)
            for cat in seen_cats:
                prompt_counts[cat] += 1

    rows = []
    for cat in category_names:
        fp = prompt_counts[cat] / n_prompts_total if n_prompts_total else 0.0
        fm = modifier_counts[cat] / n_modifiers_total if n_modifiers_total else 0.0
        rows.append({
            "Category": cat,
            "N_prompts_with_category": prompt_counts[cat],
            "f_p (prompt perspective)": round(fp, 4),
            "N_modifier_instances": modifier_counts[cat],
            "f_m (category perspective)": round(fm, 4),
        })

    df = pd.DataFrame(rows).sort_values("f_p (prompt perspective)", ascending=False)
    out_path = Path(OUTPUT_DIR) / "frequency_stats.csv"
    df.to_csv(out_path, index=False)
    print(f"[4.2] Frequency stats saved → {out_path}")
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Section 4.2 LLM classification pipeline")
    parser.add_argument(
        "--mode",
        choices=["classify", "validate", "frequency"],
        required=True,
        help=(
            "classify: run LLM on full 80K dataset; "
            "validate: evaluate against gold annotations; "
            "frequency: compute frequency stats from classified output"
        ),
    )
    parser.add_argument("--gold_path", type=str, default=None,
                        help="Path to gold annotations JSON (required for --mode validate)")
    parser.add_argument("--no_resume", action="store_true",
                        help="Start classification from scratch (ignore existing output)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(TAXONOMY_PATH):
        raise FileNotFoundError(
            f"Unified taxonomy not found at {TAXONOMY_PATH}. "
            "Run section_4_1_harmonization.py first."
        )
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    print(f"[4.2] Loaded taxonomy with {len(taxonomy)} unified categories: "
          f"{list(taxonomy.keys())}")

    if args.mode == "classify":
        run_classification(taxonomy, resume=not args.no_resume)

    elif args.mode == "validate":
        if args.gold_path is None:
            raise ValueError("--gold_path is required for --mode validate")
        run_validation(args.gold_path, taxonomy)

    elif args.mode == "frequency":
        df = compute_frequency_stats(taxonomy)
        print("\nFrequency statistics (Table 1 draft):")
        print(df.to_string(index=False))


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    main()

"""
Section 4.2 — LLM-based Modifier Classification (vLLM backend)
===============================================================
Drop-in replacement for section_4_2_classification.py that uses vLLM for
3-10x faster inference via continuous batching + prefix caching.

All 70K prompts share the same system prompt → prefix caching means the
system-prompt KV cache is computed only once.

Usage
-----
# Classify full dataset (resumes automatically)
python section_4_2_classification_vllm.py --mode classify

# Validate against gold annotations
python section_4_2_classification_vllm.py --mode validate \
    --gold_path data/gold_annotations_150.json
"""

import argparse
import ast
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ── Config ────────────────────────────────────────────────────────────────────
TAXONOMY_PATH  = "outputs/unified_taxonomy.json"
OUTPUT_DIR     = "outputs/4_2"
LLM_MODEL      = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 512
KAPPA_THRESHOLD = 0.75
DATASET_NAME   = "Gustavosta/Stable-Diffusion-Prompts"
CHUNK_SIZE     = 1000   # write results every N prompts

_token_path = os.path.expanduser("~/.cache/huggingface/token")
HF_TOKEN = open(_token_path).read().strip() if os.path.exists(_token_path) else os.environ.get("HF_TOKEN")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load validated prompt from classifier_prompt.py ──────────────────────────
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


# ── vLLM Classification pipeline ─────────────────────────────────────────────

def build_conversations(prompts: list[str]) -> list[list[dict]]:
    """Build chat messages for a list of prompt texts."""
    return [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(prompt=p)},
        ]
        for p in prompts
    ]


def run_classification(taxonomy: dict, resume: bool = True) -> None:
    import datetime
    out_path  = Path(OUTPUT_DIR) / "classified_modifiers.jsonl"
    report_path = Path(OUTPUT_DIR) / "classification_report.txt"
    LOG_EVERY = 5000   # print progress every N prompts

    # Find already-processed prompt indices
    done_ids: set[int] = set()
    if resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_ids.add(rec["prompt_id"])
                except json.JSONDecodeError:
                    pass
        print(f"[4.2] Resuming — {len(done_ids)} prompts already classified.")

    print("[4.2] Loading dataset …")
    ds = load_dataset(DATASET_NAME, split="train")
    total = len(ds)

    # Collect all pending (idx, prompt_text) pairs
    pending = [
        (idx, ex["Prompt"])
        for idx, ex in enumerate(ds)
        if idx not in done_ids
    ]
    print(f"[4.2] {len(pending)} prompts to classify.")

    if not pending:
        print("[4.2] Nothing to do.")
        return

    # ── Load vLLM engine ──────────────────────────────────────────────────────
    print("[4.2] Loading vLLM engine …")
    llm = LLM(
        model=LLM_MODEL,
        dtype="float16",
        enable_prefix_caching=True,   # system prompt KV cache reused across all prompts
        max_model_len=4096,
        tokenizer_mode="auto",
        trust_remote_code=False,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=MAX_NEW_TOKENS,
    )

    # ── Stats accumulators ────────────────────────────────────────────────────
    from collections import Counter
    cat_counter: Counter = Counter()
    # track per-prompt modifier counts for distribution
    prompts_with_cat: Counter = Counter()   # how many prompts contain each cat
    empty_count = 0
    total_mods  = 0
    processed   = 0
    start_time  = datetime.datetime.now()

    # ── Process in chunks ─────────────────────────────────────────────────────
    with open(out_path, "a") as fout:
        for chunk_start in tqdm(range(0, len(pending), CHUNK_SIZE),
                                desc="Chunks", unit="chunk"):
            chunk    = pending[chunk_start : chunk_start + CHUNK_SIZE]
            indices  = [item[0] for item in chunk]
            texts    = [item[1] for item in chunk]

            conversations = build_conversations(texts)
            outputs = llm.chat(conversations, sampling_params=sampling_params,
                               use_tqdm=False)

            for idx, prompt_text, out in zip(indices, texts, outputs):
                raw = out.outputs[0].text.strip()
                modifiers = parse_response(raw, prompt_text)
                if modifiers is None:
                    modifiers = []
                record = {
                    "prompt_id": idx,
                    "prompt":    prompt_text,
                    "modifiers": modifiers,
                }
                fout.write(json.dumps(record) + "\n")

                if not modifiers:
                    empty_count += 1
                total_mods += len(modifiers)
                cats_seen = set()
                for m in modifiers:
                    cat = m.get("category", "Unknown")
                    cat_counter[cat] += 1
                    cats_seen.add(cat)
                for cat in cats_seen:
                    prompts_with_cat[cat] += 1
                processed += 1

                # ── Every 5000 prompts: print to stdout only ──────────────────
                if processed % LOG_EVERY == 0:
                    elapsed = (datetime.datetime.now() - start_time).total_seconds()
                    done_total = len(done_ids) + processed
                    pct   = done_total / total * 100
                    speed = processed / elapsed if elapsed > 0 else 0
                    eta_s = (len(pending) - processed) / speed if speed > 0 else 0
                    eta   = str(datetime.timedelta(seconds=int(eta_s)))
                    print(
                        f"[4.2] {done_total:,}/{total:,} ({pct:.1f}%) | "
                        f"{speed:.1f} prompts/s | ETA {eta} | "
                        f"avg {total_mods/processed:.1f} mods/prompt | "
                        f"empty {empty_count/processed*100:.1f}%",
                        flush=True,
                    )

            fout.flush()

    # ── Write final report txt (same style as ollama_20_report.txt) ───────────
    # Re-read the full jsonl so the report covers ALL prompts (including resumed ones)
    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    BAR = 20
    all_cat_counter: Counter = Counter()
    all_prompts_with_cat: Counter = Counter()
    all_empty = 0
    all_mods  = 0
    all_total = 0
    with open(out_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            mods = rec.get("modifiers", [])
            all_total += 1
            all_mods  += len(mods)
            if not mods:
                all_empty += 1
            cats_seen = set()
            for m in mods:
                cat = m.get("category", "Unknown")
                all_cat_counter[cat] += 1
                cats_seen.add(cat)
            for cat in cats_seen:
                all_prompts_with_cat[cat] += 1

    avg_mods = all_mods / all_total if all_total else 0

    lines = [
        "══════════════════════════════════════════════════════════════════════",
        f"SUMMARY  (model={LLM_MODEL.split('/')[-1]}, n={all_total:,})",
        "──────────────────────────────────────────────────────────────────────",
        f"{'Category':<14} {'Prompts':>8}  {'Rate':>6}",
    ]
    for cat, cnt in all_cat_counter.most_common():
        n_prompts = all_prompts_with_cat[cat]
        rate = n_prompts / all_total if all_total else 0
        bar  = "█" * int(rate * BAR) + "░" * (BAR - int(rate * BAR))
        lines.append(f"  {cat:<12} {n_prompts:>8}  {rate*100:>5.1f}%  {bar}")

    lines += [
        "──────────────────────────────────────────────────────────────────────",
        f"Total prompts   : {all_total:,}",
        f"Total modifiers : {all_mods:,}",
        f"Avg per prompt  : {avg_mods:.2f}",
        f"Empty outputs   : {all_empty} ({all_empty/all_total*100:.1f}%)",
        f"This run time   : {str(datetime.timedelta(seconds=int(elapsed)))}",
        "══════════════════════════════════════════════════════════════════════",
    ]

    report_txt = "\n".join(lines)
    print(report_txt)
    with open(report_path, "w") as rf:
        rf.write(report_txt + "\n")

    print(f"[4.2] JSONL  → {out_path}")
    print(f"[4.2] Report → {report_path}")


# ── Validation pipeline ───────────────────────────────────────────────────────
# (identical to section_4_2_classification.py — copy-pasted for self-containment)

def run_validation(gold_path: str, taxonomy: dict) -> None:
    category_names = list(taxonomy.keys())
    out_path = Path(OUTPUT_DIR) / "classified_modifiers.jsonl"

    # Load model predictions indexed by prompt_id
    predictions: dict[int, list[dict]] = {}
    with open(out_path) as f:
        for line in f:
            rec = json.loads(line)
            predictions[rec["prompt_id"]] = rec["modifiers"]

    with open(gold_path) as f:
        gold_data = json.load(f)

    y_true, y_pred = [], []
    for item in gold_data:
        pid = item["prompt_id"]
        if pid not in predictions:
            continue
        gold_spans = {m["span"].lower(): m["category"] for m in item["modifiers"]}
        pred_spans = {m["span"].lower(): m["category"] for m in predictions[pid]}
        all_spans = set(gold_spans) | set(pred_spans)
        for span in all_spans:
            y_true.append(gold_spans.get(span, "None"))
            y_pred.append(pred_spans.get(span, "None"))

    cats_with_none = category_names + ["None"]
    kappa = cohen_kappa_score(y_true, y_pred, labels=cats_with_none)
    print(f"\nCohen's Kappa: {kappa:.4f}  (threshold ≥ {KAPPA_THRESHOLD})")
    print(classification_report(y_true, y_pred, labels=category_names, zero_division=0))

    # Save CSV
    report = classification_report(
        y_true, y_pred, labels=category_names, zero_division=0, output_dict=True
    )
    rows = []
    for cat in category_names:
        r = report.get(cat, {})
        rows.append({
            "Category": cat,
            "Precision": round(r.get("precision", 0), 3),
            "Recall":    round(r.get("recall",    0), 3),
            "F1":        round(r.get("f1-score",  0), 3),
            "Support":   int(r.get("support",     0)),
        })
    df = pd.DataFrame(rows)
    csv_path = Path(OUTPUT_DIR) / "table_kappa_f1.csv"
    df.to_csv(csv_path, index=False)
    print(f"Table saved → {csv_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=category_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=category_names,
                yticklabels=category_names, ax=ax, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    fig.tight_layout()
    pdf_path = Path(OUTPUT_DIR) / "confusion_matrix.pdf"
    fig.savefig(pdf_path)
    print(f"Confusion matrix saved → {pdf_path}")

    result = {
        "kappa": kappa,
        "pass": kappa >= KAPPA_THRESHOLD,
        "per_category": {
            cat: report.get(cat, {}) for cat in category_names
        },
    }
    val_path = Path(OUTPUT_DIR) / "validation_results.json"
    with open(val_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Validation results → {val_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["classify", "validate"], required=True)
    parser.add_argument("--gold_path", default="data/gold_annotations_150.json")
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    taxonomy = load_taxonomy(TAXONOMY_PATH)
    print(f"[4.2] Loaded taxonomy with {len(taxonomy)} unified categories: "
          f"{list(taxonomy.keys())}")

    if args.mode == "classify":
        run_classification(taxonomy, resume=not args.no_resume)
    else:
        run_validation(args.gold_path, taxonomy)


if __name__ == "__main__":
    main()

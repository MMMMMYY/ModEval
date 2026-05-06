"""
rerun_empty.py
==============
Re-runs classification for prompts that returned empty modifiers,
using a larger max_tokens budget to fix truncated JSON outputs.

Updates classified_modifiers.jsonl in-place.

Usage
-----
python rerun_empty.py
"""

import ast
import json
import os
import shutil
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ── Config ────────────────────────────────────────────────────────────────────
JSONL_PATH    = "outputs/4_2/classified_modifiers.jsonl"
DATASET_NAME  = "Gustavosta/Stable-Diffusion-Prompts"
LLM_MODEL     = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 1024          # double the original budget
CHUNK_SIZE     = 500

_token_path = os.path.expanduser("~/.cache/huggingface/token")
HF_TOKEN = open(_token_path).read().strip() if os.path.exists(_token_path) else os.environ.get("HF_TOKEN")

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


def main():
    jsonl_path = Path(JSONL_PATH)

    # ── Step 1: load all existing records ────────────────────────────────────
    print("Loading existing results …")
    records: dict[int, dict] = {}
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                records[rec["prompt_id"]] = rec
            except json.JSONDecodeError:
                pass

    empty_ids = {pid for pid, rec in records.items() if not rec.get("modifiers")}
    print(f"Found {len(empty_ids):,} empty records out of {len(records):,} total.")

    if not empty_ids:
        print("Nothing to re-run.")
        return

    # ── Step 2: get prompt texts from dataset ─────────────────────────────────
    print("Loading dataset …")
    ds = load_dataset(DATASET_NAME, split="train")
    pending = [
        (idx, ex["Prompt"])
        for idx, ex in enumerate(ds)
        if idx in empty_ids
    ]
    print(f"Re-running {len(pending):,} prompts with max_tokens={MAX_NEW_TOKENS} …")

    # ── Step 3: load vLLM ─────────────────────────────────────────────────────
    llm = LLM(
        model=LLM_MODEL,
        dtype="float16",
        enable_prefix_caching=True,
        max_model_len=4096,
        trust_remote_code=False,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)

    fixed = 0
    still_empty = 0

    # ── Step 4: classify in chunks ────────────────────────────────────────────
    for chunk_start in tqdm(range(0, len(pending), CHUNK_SIZE), desc="Chunks"):
        chunk   = pending[chunk_start : chunk_start + CHUNK_SIZE]
        indices = [item[0] for item in chunk]
        texts   = [item[1] for item in chunk]

        conversations = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(prompt=p)},
            ]
            for p in texts
        ]
        outputs = llm.chat(conversations, sampling_params=sampling_params, use_tqdm=False)

        for idx, prompt_text, out in zip(indices, texts, outputs):
            raw = out.outputs[0].text.strip()
            modifiers = parse_response(raw, prompt_text)
            if modifiers is None:
                modifiers = []
            records[idx]["modifiers"] = modifiers
            if modifiers:
                fixed += 1
            else:
                still_empty += 1

    print(f"\nFixed  : {fixed:,}  (was empty, now has modifiers)")
    print(f"Still empty: {still_empty:,}")

    # ── Step 5: write updated jsonl ───────────────────────────────────────────
    backup = jsonl_path.with_suffix(".jsonl.bak")
    shutil.copy(jsonl_path, backup)
    print(f"Backup saved → {backup}")

    with open(jsonl_path, "w") as fout:
        for rec in sorted(records.values(), key=lambda r: r["prompt_id"]):
            fout.write(json.dumps(rec) + "\n")

    print(f"Updated → {jsonl_path}")

    # ── Step 6: print updated summary ─────────────────────────────────────────
    from collections import Counter
    cat_counter: Counter = Counter()
    prompts_with_cat: Counter = Counter()
    all_empty = 0
    all_mods  = 0
    for rec in records.values():
        mods = rec.get("modifiers", [])
        all_mods += len(mods)
        if not mods:
            all_empty += 1
        cats_seen = set()
        for m in mods:
            cat = m.get("category", "Unknown")
            cat_counter[cat] += 1
            cats_seen.add(cat)
        for cat in cats_seen:
            prompts_with_cat[cat] += 1

    total = len(records)
    avg   = all_mods / total if total else 0
    BAR   = 20
    lines = [
        "══════════════════════════════════════════════════════════════════════",
        f"UPDATED SUMMARY  (n={total:,})",
        "──────────────────────────────────────────────────────────────────────",
        f"{'Category':<14} {'Prompts':>8}  {'Rate':>6}",
    ]
    for cat, _ in cat_counter.most_common():
        n = prompts_with_cat[cat]
        rate = n / total
        bar  = "█" * int(rate * BAR) + "░" * (BAR - int(rate * BAR))
        lines.append(f"  {cat:<12} {n:>8}  {rate*100:>5.1f}%  {bar}")
    lines += [
        "──────────────────────────────────────────────────────────────────────",
        f"Total prompts   : {total:,}",
        f"Total modifiers : {all_mods:,}",
        f"Avg per prompt  : {avg:.2f}",
        f"Empty outputs   : {all_empty} ({all_empty/total*100:.1f}%)",
        "══════════════════════════════════════════════════════════════════════",
    ]
    report = "\n".join(lines)
    print(report)

    report_path = Path("outputs/4_2/classification_report.txt")
    with open(report_path, "a") as rf:
        rf.write("\n\n" + report + "\n")
    print(f"Report appended → {report_path}")


if __name__ == "__main__":
    main()

"""
compute_kappa.py
================
Computes inter-annotator agreement (IAA) and LLM validation metrics.

Steps:
  1. Parse 3 annotation files (two different formats)
  2. Compute pairwise Cohen's Kappa between annotators (IAA)
  3. Build gold standard via majority voting
  4. Compute LLM vs gold Kappa + per-category F1
  5. Write results to outputs/4_2/kappa_results.txt and table_kappa_f1.csv
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, classification_report

OUTPUT_DIR   = Path("outputs/4_2")
JSONL_PATH   = OUTPUT_DIR / "classified_modifiers.jsonl"
GOLD_JSONL   = OUTPUT_DIR / "gold_annotation_150.jsonl"
ANNO_FILES   = [
    OUTPUT_DIR / "annotation_1_tuned.txt",
    OUTPUT_DIR / "annotation_2_tuned.txt",
    OUTPUT_DIR / "annotation_3_tuned.txt",
]
CATS = ["Artist", "Medium", "Movement", "Trending",
        "Quality", "Atmosphere", "Repeating", "Magic"]


# ── Parsers ───────────────────────────────────────────────────────────────────

def norm(span: str) -> str:
    """Normalise a span for matching: lowercase, collapse whitespace, strip punctuation edges."""
    s = span.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    s = s.strip('.,;:!?"\'-')
    return s


def parse_inline(text: str) -> dict[int, dict[str, list[str]]]:
    """Parse annotation_1.txt inline format:
    Artist: [span1, span2] Medium: [span1] ...
    """
    results: dict[int, dict[str, list[str]]] = {}
    # Split into prompt blocks
    blocks = re.split(r'──\s*\[\d+\]\s*\(id=(\d+)\)', text)
    # blocks: [before_first, id1, content1, id2, content2, ...]
    i = 1
    while i < len(blocks) - 1:
        pid = int(blocks[i])
        content = blocks[i + 1]
        annotation_line = re.search(r'YOUR ANNOTATION:\s*(.+?)(?=──|\Z)', content, re.DOTALL)
        if not annotation_line:
            i += 2
            continue
        anno_text = annotation_line.group(1).strip()
        anno: dict[str, list[str]] = {c: [] for c in CATS}
        for cat in CATS:
            m = re.search(rf'{cat}:\s*\[([^\]]*)\]', anno_text)
            if m:
                raw = m.group(1).strip()
                if raw:
                    spans = [s.strip() for s in raw.split(',') if s.strip()]
                    anno[cat] = [norm(s) for s in spans]
        results[pid] = anno
        i += 2
    return results


def parse_json_format(text: str) -> dict[int, dict[str, list[str]]]:
    """Parse annotation_2/3.txt JSON-per-prompt format."""
    results: dict[int, dict[str, list[str]]] = {}
    blocks = re.split(r'──\s*\[\d+\]\s*\(id=(\d+)\)', text)
    i = 1
    while i < len(blocks) - 1:
        pid = int(blocks[i])
        content = blocks[i + 1]
        # Extract JSON block
        json_m = re.search(r'\{[\s\S]*?\}', content)
        if json_m:
            try:
                obj = json.loads(json_m.group(0))
                anno: dict[str, list[str]] = {c: [] for c in CATS}
                for cat in CATS:
                    raw = obj.get(cat, [])
                    if isinstance(raw, list):
                        anno[cat] = [norm(s) for s in raw if s.strip()]
                    elif isinstance(raw, str) and raw.strip():
                        anno[cat] = [norm(raw)]
                results[pid] = anno
            except json.JSONDecodeError:
                pass
        i += 2
    return results


def parse_annotation_file(path: Path) -> dict[int, dict[str, list[str]]]:
    text = path.read_text(encoding='utf-8')
    # Detect format by checking if JSON blocks are present
    if re.search(r'YOUR ANNOTATION:\s*\n\s*\{', text):
        return parse_json_format(text)
    else:
        return parse_inline(text)


# ── Span-level comparison ─────────────────────────────────────────────────────

def anno_to_span_dict(anno: dict[str, list[str]]) -> dict[str, str]:
    """Flatten annotation to {norm_span: category}."""
    result = {}
    for cat, spans in anno.items():
        for s in spans:
            if s:
                result[s] = cat
    return result


def build_comparison_vectors(
    anno_a: dict[str, list[str]],
    anno_b: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    """Build label vectors for pairwise Kappa over union of spans."""
    sd_a = anno_to_span_dict(anno_a)
    sd_b = anno_to_span_dict(anno_b)
    all_spans = set(sd_a) | set(sd_b)
    labels_a, labels_b = [], []
    for sp in all_spans:
        labels_a.append(sd_a.get(sp, "None"))
        labels_b.append(sd_b.get(sp, "None"))
    return labels_a, labels_b


def pairwise_kappa(
    all_annos: list[dict[int, dict[str, list[str]]]],
    prompt_ids: list[int],
) -> dict[str, float]:
    """Compute all 3 pairwise Kappa scores."""
    pairs = [(0, 1, "A-B"), (0, 2, "A-C"), (1, 2, "B-C")]
    kappas = {}
    for i, j, name in pairs:
        ya, yb = [], []
        for pid in prompt_ids:
            a = all_annos[i].get(pid, {c: [] for c in CATS})
            b = all_annos[j].get(pid, {c: [] for c in CATS})
            la, lb = build_comparison_vectors(a, b)
            ya.extend(la)
            yb.extend(lb)
        all_labels = sorted(set(ya) | set(yb))
        k = cohen_kappa_score(ya, yb, labels=all_labels)
        kappas[name] = k
    return kappas


# ── Gold standard via majority vote ──────────────────────────────────────────

def majority_vote_gold(
    all_annos: list[dict[int, dict[str, list[str]]]],
    prompt_ids: list[int],
) -> dict[int, dict[str, list[str]]]:
    """For each span in the union of all 3 annotators, take majority-vote category."""
    gold: dict[int, dict[str, list[str]]] = {}
    for pid in prompt_ids:
        annos = [a.get(pid, {c: [] for c in CATS}) for a in all_annos]
        span_votes: dict[str, Counter] = defaultdict(Counter)
        for anno in annos:
            for cat, spans in anno.items():
                for sp in spans:
                    if sp:
                        span_votes[sp][cat] += 1
        # Take majority (≥2/3 agree); if tie, take alphabetically first
        gold_anno: dict[str, list[str]] = {c: [] for c in CATS}
        for sp, votes in span_votes.items():
            top_cat, top_cnt = votes.most_common(1)[0]
            if top_cnt >= 2:   # at least 2/3 agree
                gold_anno[top_cat].append(sp)
        gold[pid] = gold_anno
    return gold


# ── LLM vs Gold ───────────────────────────────────────────────────────────────

def load_llm_predictions(jsonl_path: Path, prompt_ids: set[int]) -> dict[int, dict[str, list[str]]]:
    preds: dict[int, dict[str, list[str]]] = {}
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec["prompt_id"] not in prompt_ids:
                continue
            anno: dict[str, list[str]] = {c: [] for c in CATS}
            for m in rec.get("modifiers", []):
                cat = m.get("category")
                sp  = norm(m.get("span", ""))
                if cat in CATS and sp:
                    anno[cat].append(sp)
            preds[rec["prompt_id"]] = anno
    return preds


def llm_vs_gold_metrics(
    gold: dict[int, dict[str, list[str]]],
    llm:  dict[int, dict[str, list[str]]],
    prompt_ids: list[int],
) -> tuple[float, dict]:
    y_gold, y_llm = [], []
    for pid in prompt_ids:
        g = gold.get(pid, {c: [] for c in CATS})
        p = llm.get(pid,  {c: [] for c in CATS})
        lg, lp = build_comparison_vectors(g, p)
        y_gold.extend(lg)
        y_llm.extend(lp)
    all_labels = sorted(set(y_gold) | set(y_llm))
    kappa = cohen_kappa_score(y_gold, y_llm, labels=all_labels)
    report = classification_report(
        y_gold, y_llm, labels=CATS, zero_division=0, output_dict=True
    )
    return kappa, report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load annotations
    print("Parsing annotation files …")
    all_annos = []
    for path in ANNO_FILES:
        anno = parse_annotation_file(path)
        all_annos.append(anno)
        print(f"  {path.name}: {len(anno)} prompts parsed")

    # Common prompt IDs across all 3 annotators
    common_ids = sorted(
        set(all_annos[0]) & set(all_annos[1]) & set(all_annos[2])
    )
    print(f"Common prompts across all 3 annotators: {len(common_ids)}")

    # ── IAA Kappa ─────────────────────────────────────────────────────────────
    print("\nComputing pairwise IAA Kappa …")
    iaa = pairwise_kappa(all_annos, common_ids)
    iaa_mean = sum(iaa.values()) / len(iaa)
    for pair, k in iaa.items():
        print(f"  κ({pair}) = {k:.4f}")
    print(f"  κ(mean)  = {iaa_mean:.4f}  (threshold ≥ 0.75)")

    # ── Gold standard ─────────────────────────────────────────────────────────
    print("\nBuilding gold standard via majority vote …")
    gold = majority_vote_gold(all_annos, common_ids)
    total_gold_spans = sum(
        sum(len(v) for v in g.values()) for g in gold.values()
    )
    print(f"  Gold spans total: {total_gold_spans}")

    # ── LLM vs Gold ───────────────────────────────────────────────────────────
    print("\nComputing LLM vs Gold metrics …")
    llm = load_llm_predictions(JSONL_PATH, set(common_ids))
    print(f"  LLM predictions loaded: {len(llm)} prompts")
    llm_kappa, report = llm_vs_gold_metrics(gold, llm, common_ids)
    print(f"  κ(LLM vs Gold) = {llm_kappa:.4f}  (threshold ≥ 0.75)")

    # ── Per-category F1 table ─────────────────────────────────────────────────
    rows = []
    for cat in CATS:
        r = report.get(cat, {})
        rows.append({
            "Category":  cat,
            "Precision": round(r.get("precision", 0), 3),
            "Recall":    round(r.get("recall",    0), 3),
            "F1":        round(r.get("f1-score",  0), 3),
            "Support":   int(r.get("support",     0)),
        })
    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "table_kappa_f1.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nPer-category F1 table saved → {csv_path}")

    # ── Text report ───────────────────────────────────────────────────────────
    BAR = 20
    lines = [
        "══════════════════════════════════════════════════════════════════════",
        "SECTION 4.2 VALIDATION RESULTS",
        "══════════════════════════════════════════════════════════════════════",
        "",
        "── Inter-Annotator Agreement (IAA) ───────────────────────────────────",
        f"  κ(A-B)  = {iaa['A-B']:.4f}",
        f"  κ(A-C)  = {iaa['A-C']:.4f}",
        f"  κ(B-C)  = {iaa['B-C']:.4f}",
        f"  κ(mean) = {iaa_mean:.4f}  ({'✓ PASS' if iaa_mean >= 0.75 else '✗ FAIL'} threshold ≥ 0.75)",
        "",
        "── LLM vs Gold Standard ──────────────────────────────────────────────",
        f"  κ(LLM vs Gold) = {llm_kappa:.4f}  ({'✓ PASS' if llm_kappa >= 0.75 else '✗ FAIL'} threshold ≥ 0.75)",
        f"  Prompts evaluated: {len(common_ids)}",
        "",
        "── Per-Category F1 ───────────────────────────────────────────────────",
        f"  {'Category':<12} {'P':>6} {'R':>6} {'F1':>6} {'Support':>8}",
        "  " + "─" * 42,
    ]
    for _, row in df.iterrows():
        bar = "█" * int(row["F1"] * BAR) + "░" * (BAR - int(row["F1"] * BAR))
        lines.append(
            f"  {row['Category']:<12} {row['Precision']:>6.3f} {row['Recall']:>6.3f} "
            f"{row['F1']:>6.3f} {int(row['Support']):>8}  {bar}"
        )
    lines += [
        "  " + "─" * 42,
        f"  {'Macro avg':<12} "
        f"{report['macro avg']['precision']:>6.3f} "
        f"{report['macro avg']['recall']:>6.3f} "
        f"{report['macro avg']['f1-score']:>6.3f}",
        "",
        "══════════════════════════════════════════════════════════════════════",
    ]

    report_txt = "\n".join(lines)
    print("\n" + report_txt)

    result_path = OUTPUT_DIR / "kappa_results.txt"
    result_path.write_text(report_txt + "\n")
    print(f"\nFull report → {result_path}")
    print(f"CSV table   → {csv_path}")


if __name__ == "__main__":
    main()

"""
Section 4.3 — Steering Modifier Identification
===============================================
Step 1: Frequency-based candidate selection  (--mode freq)
Step 2: Dual-metric influence validation
        --mode sample     sample 25 prompts for image experiments
        --mode generate   generate images with SD3 (A6000 required)
        --mode evaluate   compute SSIM / LPIPS / MSCS
        --mode report     statistical tests + final tables

Outputs (all under outputs/4_3/)
---------------------------------
  table1_frequency.csv         Table 1: f_p and f_m per category
  candidates.json              list of candidate categories
  sampled_25_prompts.jsonl     25 prompts + subprompts
  sampled_25_prompts.txt       human-readable preview
  images/                      generated images per condition
  metrics.json                 raw SSIM / LPIPS / MSCS per prompt
  table2_image_metrics.csv     Table 2: mean±std per condition/metric
  table3_mscs.csv              MSCS per category + t-test
  steering_report.txt          final narrative summary

Usage
-----
python section_4_3_steering.py --mode freq
python section_4_3_steering.py --mode sample
python section_4_3_steering.py --mode generate   # on A6000
python section_4_3_steering.py --mode evaluate   # on A6000
python section_4_3_steering.py --mode report
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
JSONL_PATH   = "outputs/4_2/classified_modifiers.jsonl"
OUTPUT_DIR   = Path("outputs/4_3")
DATASET_NAME = "Gustavosta/Stable-Diffusion-Prompts"
SEED         = 42

CATS = ["Artist", "Medium", "Movement", "Trending",
        "Quality", "Atmosphere", "Repeating", "Magic"]

# f_p threshold for candidate steering modifiers
FREQ_THRESHOLD = 0.20

# Image generation settings
N_SAMPLE_PROMPTS       = 25
N_IMAGES_PER_SUBPROMPT = 4
N_TARGET_IMAGES        = 2
N_TARGETED_PROMPTS     = 10   # for Repeating / Magic targeted sampling
TARGETED_CATS          = ["Repeating", "Magic"]   # low-prevalence, targeted design

# Two T2I models — generated via HF Inference API (no local weights needed)
# FLUX.1-schnell: open-weight, free, 4-step, paper cites as "FLUX.1-schnell"
MODELS = {
    "flux":  "black-forest-labs/FLUX.1-schnell",
    "sdxl":  "stabilityai/stable-diffusion-xl-base-1.0",
}

SUBJECT_TYPES = ["portrait", "landscape", "character", "object", "abstract"]
N_PER_TYPE    = N_SAMPLE_PROMPTS // len(SUBJECT_TYPES)  # 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_records(path: str = JSONL_PATH) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def get_cat_spans(rec: dict) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {c: [] for c in CATS}
    for m in rec.get("modifiers", []):
        cat = m.get("category")
        if cat in CATS:
            result[cat].append(m)
    return result


_STOP_WORDS = {
    'a', 'an', 'the', 'of', 'in', 'with', 'as', 'and', 'or', 'by', 'for',
    'to', 'on', 'at', 'from', 'but', 'is', 'are', 'was', 'were', 'be',
    'been', 'has', 'have', 'had', 'do', 'does', 'did', 'its', 'it', 'this',
    'that', 'these', 'those', 'into', 'up', 'out', 'over', 'under', 'between',
    'some', 'not', 'no', 'so', 'if', 'than', 'then', 'about', 'which',
}

def content_word_count(text: str) -> int:
    """Count non-stop-word alphabetic tokens."""
    words = re.findall(r'[a-zA-Z]+', text.lower())
    return sum(1 for w in words if w not in _STOP_WORDS)


def extract_subject(prompt: str, modifiers: list[dict]) -> str:
    """Remove all modifier spans from prompt to get subject tokens."""
    spans = sorted(modifiers, key=lambda m: m.get("start", 0), reverse=True)
    text = prompt
    for m in spans:
        s, e = m.get("start"), m.get("end")
        if s is not None and e is not None and 0 <= s < e <= len(text):
            text = text[:s] + text[e:]
    # Split on commas, semicolons, pipes — keep only segments with real words
    parts = [p.strip() for p in re.split(r'[,;|]', text)]
    parts = [p for p in parts if re.search(r'[a-zA-Z0-9]', p)]
    text = ', '.join(parts)
    text = re.sub(r'\s+', ' ', text).strip().strip('.,;: ')
    return text


def build_subprompts(rec: dict, candidates: list[str]) -> dict[str, str]:
    prompt   = rec["prompt"]
    all_mods = rec.get("modifiers", [])
    by_cat   = get_cat_spans(rec)
    subject  = extract_subject(prompt, all_mods)

    result = {"subject": subject}

    # Subj + c for each candidate
    for cat in candidates:
        spans_text = ", ".join(m["span"] for m in by_cat[cat])
        result[f"subj+{cat}"] = (subject + ", " + spans_text).strip(", ")

    # Subj + All (all candidate modifiers)
    cand_spans = [m["span"] for cat in candidates for m in by_cat[cat]]
    result["subj+all"] = (subject + ", " + ", ".join(cand_spans)).strip(", ")

    return result


# ── Subject-type keyword classifier ──────────────────────────────────────────

_SUBJECT_KW = {
    "portrait":  ["portrait", "face", "headshot", "bust", "close-up", "close up",
                  "girl", "woman", "man", "boy", "person", "human", "female", "male"],
    "landscape": ["landscape", "scenery", "nature", "forest", "mountain", "ocean",
                  "sea", "sky", "field", "valley", "desert", "beach", "lake",
                  "environment", "background"],
    "character": ["character", "warrior", "wizard", "knight", "hero", "villain",
                  "adventurer", "paladin", "mage", "rogue", "ranger", "druid",
                  "elf", "dwarf", "orc", "dragon", "creature", "monster"],
    "object":    ["car", "weapon", "sword", "gun", "building", "architecture",
                  "vehicle", "robot", "machine", "device", "artifact", "statue",
                  "sculpture", "ship", "spaceship"],
    "abstract":  ["abstract", "geometric", "pattern", "fractal", "surreal",
                  "dreamlike", "psychedelic", "cosmic", "mystical", "3d render"],
}

def classify_subject_type(prompt: str) -> str:
    p = prompt.lower()
    scores = {t: sum(1 for kw in kws if kw in p) for t, kws in _SUBJECT_KW.items()}
    best = max(scores, key=lambda t: scores[t])
    return best if scores[best] > 0 else "abstract"


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: FREQUENCY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_freq():
    print("[4.3] Loading records …")
    records = load_records()
    total_prompts = len(records)

    prompt_count: Counter  = Counter()
    modifier_count: Counter = Counter()
    total_mods = 0

    for rec in records:
        cats_seen = set()
        for m in rec.get("modifiers", []):
            cat = m.get("category")
            if cat in CATS:
                modifier_count[cat] += 1
                total_mods += 1
                cats_seen.add(cat)
        for cat in cats_seen:
            prompt_count[cat] += 1

    rows = []
    for cat in CATS:
        fp = prompt_count[cat] / total_prompts
        fm = modifier_count[cat] / total_mods if total_mods else 0
        rows.append({
            "Category":    cat,
            "N_prompts":   prompt_count[cat],
            "f_p (%)":     round(fp * 100, 2),
            "N_modifiers": modifier_count[cat],
            "f_m (%)":     round(fm * 100, 2),
            "Candidate":   "✓" if fp > FREQ_THRESHOLD else "✗",
        })

    df = pd.DataFrame(rows).sort_values("f_p (%)", ascending=False)
    csv_path = OUTPUT_DIR / "table1_frequency.csv"
    df.to_csv(csv_path, index=False)

    candidates = [r["Category"] for _, r in df.iterrows() if r["Candidate"] == "✓"]
    (OUTPUT_DIR / "candidates.json").write_text(json.dumps(candidates))

    # Print Table 1
    print(f"\n[4.3] Total prompts : {total_prompts:,}")
    print(f"[4.3] Total modifiers: {total_mods:,}")
    print(f"\n── Table 1: Category Frequency ──────────────────────────────────────")
    print(f"  {'Category':<12} {'N_prompts':>10} {'f_p':>7}  {'N_mods':>8} {'f_m':>7}  Cand.")
    print("  " + "─" * 60)
    for _, r in df.iterrows():
        bar = "█" * int(r["f_p (%)"] / 5) + "░" * (20 - int(r["f_p (%)"] / 5))
        print(f"  {r['Category']:<12} {r['N_prompts']:>10,} {r['f_p (%)']:>6.1f}%"
              f"  {r['N_modifiers']:>8,} {r['f_m (%)']:>6.1f}%   {r['Candidate']}  {bar}")

    print(f"\n  Candidates (f_p > {FREQ_THRESHOLD*100:.0f}%): {candidates}")
    print(f"\n  Table 1 → {csv_path}")
    return candidates


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: SAMPLE 25 PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

def run_sample():
    import random
    random.seed(SEED)

    candidates_path = OUTPUT_DIR / "candidates.json"
    if not candidates_path.exists():
        print("[4.3] Run --mode freq first.")
        return
    candidates = json.loads(candidates_path.read_text())
    print(f"[4.3] Candidates: {candidates}")

    print("[4.3] Loading records …")
    records = load_records()

    # Must contain ≥1 modifier from every candidate category
    # AND have a clean subject with ≥3 meaningful words
    # Manually excluded IDs (confirmed bad subjects after inspection)
    EXCLUDE_IDS = {6921, 69491}

    MIN_CONTENT_WORDS = 3  # content words (non-stop) required in subject

    # Patterns that indicate a Trending/Artist modifier leaked into subject
    _LEAK_RE = re.compile(
        r'\bartstation\b|\bdeviantart\b|\bpixiv\b|\bcgsociety\b|\bhearthstone\b'
        r'|\bart by\b|\bby [A-Z][a-z]'
        r'|\balphonse mucha\b|\bgreg rutkowski\b|\bartgerm\b|\bwlop\b'
        r'|\boctane render\b|\bunreal engine\b|\bbeeple\b',
        re.IGNORECASE,
    )

    def has_all_candidates(rec):
        by_cat = get_cat_spans(rec)
        if not all(len(by_cat[c]) > 0 for c in candidates):
            return False
        subject = extract_subject(rec["prompt"], rec.get("modifiers", []))
        # Reject manually excluded IDs
        if rec.get("prompt_id") in EXCLUDE_IDS:
            return False
        # Reject encoding-corrupt prompts
        if re.search(r'[\x80-\xff]|ï¿½', subject):
            return False
        # Reject prompts where known modifier terms leaked into subject
        if _LEAK_RE.search(subject):
            return False
        return content_word_count(subject) >= MIN_CONTENT_WORDS

    eligible = [r for r in records if has_all_candidates(r)]
    print(f"[4.3] Eligible prompts: {len(eligible):,}")

    for rec in eligible:
        rec["_subject_type"] = classify_subject_type(rec["prompt"])

    by_type: dict[str, list] = defaultdict(list)
    for rec in eligible:
        by_type[rec["_subject_type"]].append(rec)

    selected = []
    for stype in SUBJECT_TYPES:
        pool = by_type[stype]
        random.shuffle(pool)
        chosen = pool[:N_PER_TYPE]
        print(f"  {stype:<12}: {len(chosen)}/{N_PER_TYPE}  (pool {len(pool):,})")
        selected.extend(chosen)

    # Top-up to 25
    if len(selected) < N_SAMPLE_PROMPTS:
        done_ids = {r["prompt_id"] for r in selected}
        remaining = [r for r in eligible if r["prompt_id"] not in done_ids]
        random.shuffle(remaining)
        topup = remaining[:N_SAMPLE_PROMPTS - len(selected)]
        selected.extend(topup)
        print(f"  top-up      : +{len(topup)}")

    selected = selected[:N_SAMPLE_PROMPTS]

    output_records = []
    for rec in selected:
        subprompts = build_subprompts(rec, candidates)
        output_records.append({
            "prompt_id":    rec["prompt_id"],
            "prompt":       rec["prompt"],
            "subject_type": rec.get("_subject_type", "unknown"),
            "subprompts":   subprompts,
            "modifiers":    rec.get("modifiers", []),
        })

    out_path = OUTPUT_DIR / "sampled_25_prompts.jsonl"
    with open(out_path, "w") as f:
        for rec in output_records:
            f.write(json.dumps(rec) + "\n")

    # Human-readable
    txt_path = OUTPUT_DIR / "sampled_25_prompts.txt"
    with open(txt_path, "w") as f:
        f.write("25 sampled prompts for Section 4.3 image generation experiments\n")
        f.write("=" * 70 + "\n\n")
        for i, rec in enumerate(output_records, 1):
            f.write(f"── [{i:02d}] id={rec['prompt_id']}  type={rec['subject_type']}\n")
            f.write(f"ORIGINAL : {rec['prompt']}\n")
            for cond, text in rec["subprompts"].items():
                f.write(f"  {cond:<16}: {text[:100]}{'…' if len(text) > 100 else ''}\n")
            f.write("\n")

    print(f"\n[4.3] Saved → {out_path}")
    print(f"[4.3] Preview → {txt_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 3: IMAGE GENERATION (SD3, A6000 required)
# ══════════════════════════════════════════════════════════════════════════════

def _load_pipeline(model_key: str):
    """Load FLUX.1-schnell or SDXL with A6000 optimisations."""
    import torch
    model_id = MODELS[model_key]
    if model_key == "flux":
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        ).to("cuda")
    elif model_key == "sdxl":
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True
        ).to("cuda")
    else:
        raise ValueError(f"Unknown model: {model_key}")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print(f"  xformers enabled")
    except Exception:
        print(f"  xformers not available, using default attention")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_generate(model_key: str = "flux"):
    import time
    import torch

    sampled_path = OUTPUT_DIR / "sampled_25_prompts.jsonl"
    if not sampled_path.exists():
        print("[4.3] Run --mode sample first.")
        return

    candidates = json.loads((OUTPUT_DIR / "candidates.json").read_text())
    records = []
    with open(sampled_path) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"[4.3] Loading {model_key} ({MODELS[model_key]}) …")
    pipe = _load_pipeline(model_key)
    generator = torch.Generator("cuda").manual_seed(SEED)

    img_root = OUTPUT_DIR / "images" / model_key
    img_root.mkdir(parents=True, exist_ok=True)

    conditions = (
        ["subject", "subj+all", "target"]
        + [f"subj+{c}" for c in candidates]
    )
    n_per_cond = {c: (N_TARGET_IMAGES if c == "target" else N_IMAGES_PER_SUBPROMPT)
                  for c in conditions}

    # Build full task list: (prompt_text, out_path)
    # Each image is a separate task — batching across different prompts
    # maximises GPU utilisation (42 GB free on A6000 after model load)
    tasks: list[tuple[str, Path]] = []
    for rec in records:
        pid  = rec["prompt_id"]
        subs = rec["subprompts"]
        (img_root / str(pid)).mkdir(exist_ok=True)
        for cond in conditions:
            n        = n_per_cond[cond]
            cond_dir = img_root / str(pid) / cond
            cond_dir.mkdir(exist_ok=True)
            prompt_text = rec["prompt"] if cond == "target" else subs.get(cond, "")
            if not prompt_text:
                continue
            for idx in range(n):
                out = cond_dir / f"{idx:02d}.png"
                if not out.exists():
                    tasks.append((prompt_text, out))

    total = len(tasks)
    print(f"[4.3] {total} images  |  batch across different prompts for max GPU use")

    # A6000 48 GB: FLUX ~6 GB weights → batch 16 safe (~30 GB total)
    #              SDXL ~6.5 GB weights → batch 12 safe (~28 GB total)
    BATCH = 16 if model_key == "flux" else 12

    # Pipeline kwargs
    pipe_kwargs = (
        dict(num_inference_steps=4,  guidance_scale=0.0)   # FLUX.1-schnell
        if model_key == "flux" else
        dict(num_inference_steps=30, guidance_scale=7.5)   # SDXL
    )

    done = 0
    t0   = time.time()

    for i in range(0, total, BATCH):
        chunk        = tasks[i : i + BATCH]
        prompts      = [t[0] for t in chunk]
        out_paths    = [t[1] for t in chunk]

        images = pipe(
            prompt=prompts,
            num_images_per_prompt=1,
            generator=generator,
            **pipe_kwargs,
        ).images

        for img, path in zip(images, out_paths):
            img.save(path)

        done   += len(chunk)
        elapsed = time.time() - t0
        eta     = elapsed / done * (total - done) if done else 0
        print(f"  [{done}/{total}]  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m",
              flush=True)

    print(f"\n[4.3] [{model_key}] Generation complete → {img_root}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 4: EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluate(model_key: str = "flux"):
    import torch
    import clip
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim_fn
    import lpips as lpips_lib

    sampled_path = OUTPUT_DIR / "sampled_25_prompts.jsonl"
    candidates   = json.loads((OUTPUT_DIR / "candidates.json").read_text())
    img_root     = OUTPUT_DIR / "images" / model_key

    records = []
    with open(sampled_path) as f:
        for line in f:
            records.append(json.loads(line))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[4.3] Loading CLIP ViT-B/32 …")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    print("[4.3] Loading LPIPS …")
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)

    # Build CLIP text anchors T_c (top-3 most frequent modifier terms per category)
    print("[4.3] Building text anchors …")
    all_recs = load_records()
    text_anchors: dict[str, torch.Tensor] = {}
    for cat in CATS:
        counter: Counter = Counter()
        for rec in all_recs:
            for m in rec.get("modifiers", []):
                if m.get("category") == cat:
                    counter[m["span"].lower().strip()] += 1
        terms = [t for t, _ in counter.most_common(3)] or [cat.lower()]
        tokens = clip.tokenize(terms, truncate=True).to(device)
        with torch.no_grad():
            e = clip_model.encode_text(tokens).float()
            e = e / e.norm(dim=-1, keepdim=True)
            anchor = e.mean(dim=0)
            text_anchors[cat] = anchor / anchor.norm()
        print(f"  T_{cat}: {terms}")

    # Helper functions
    def load_np(folder: Path) -> list[np.ndarray]:
        return [np.array(Image.open(p).convert("RGB").resize((512, 512)))
                for p in sorted(folder.glob("*.png"))]

    def load_tensor(folder: Path) -> Optional[torch.Tensor]:
        imgs = [torch.tensor(np.array(Image.open(p).convert("RGB").resize((512, 512))))
                .permute(2, 0, 1).float() / 127.5 - 1.0
                for p in sorted(folder.glob("*.png"))]
        return torch.stack(imgs).to(device) if imgs else None

    def load_clip(folder: Path) -> Optional[torch.Tensor]:
        imgs = [clip_preprocess(Image.open(p).convert("RGB"))
                for p in sorted(folder.glob("*.png"))]
        return torch.stack(imgs).to(device) if imgs else None

    def img_embed(clip_t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            e = clip_model.encode_image(clip_t).float()
            e = e / e.norm(dim=-1, keepdim=True)
        return e.mean(dim=0)

    conditions_eval = (["subject", "subj+all"]
                       + [f"subj+{c}" for c in candidates])
    all_results: dict = {}

    for rec in records:
        pid     = rec["prompt_id"]
        pid_dir = img_root / str(pid)
        print(f"[4.3] Evaluating prompt {pid} …")

        target_dir = pid_dir / "target"
        if not target_dir.exists() or not list(target_dir.glob("*.png")):
            print(f"  WARNING: no target images for {pid}, skipping")
            continue

        target_np   = load_np(target_dir)
        target_ref  = target_np[0]
        target_clip = load_clip(target_dir)
        target_t    = load_tensor(target_dir)

        result: dict = {}
        for cond in conditions_eval:
            cond_dir = pid_dir / cond
            if not cond_dir.exists():
                continue

            cond_np = load_np(cond_dir)
            ssim_vals = [float(ssim_fn(target_ref, img, channel_axis=2,
                                       data_range=255))
                         for img in cond_np]

            cond_t = load_tensor(cond_dir)
            lpips_vals = []
            if cond_t is not None and target_t is not None:
                t_ref = target_t[:1].expand(len(cond_t), -1, -1, -1)
                with torch.no_grad():
                    lp = lpips_fn(cond_t, t_ref).squeeze().cpu().numpy()
                lpips_vals = lp.tolist() if lp.ndim > 0 else [float(lp)]

            # MSCS: CLIP cosine sim with text anchor T_c
            mscs_vals: dict[str, float] = {}
            cond_clip = load_clip(cond_dir)
            if cond_clip is not None:
                cond_embed = img_embed(cond_clip)
                for cat in CATS:
                    sim = float(torch.dot(cond_embed, text_anchors[cat]).item())
                    mscs_vals[cat] = sim

            result[cond] = {"ssim": ssim_vals, "lpips": lpips_vals, "mscs": mscs_vals}

        all_results[str(pid)] = result

    # ── FID: computed per condition across all prompts ────────────────────────
    # torchmetrics FID accumulates real/fake image feature statistics.
    # Reference (real) = target images; Fake = subprompt-generated images.
    fid_per_cond: dict[str, float] = {}
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from PIL import Image as PILImage

        def load_uint8_tensor(folder: Path) -> Optional[torch.Tensor]:
            """Return (N,3,H,W) uint8 tensor resized to 299×299 for InceptionV3."""
            imgs = []
            for p in sorted(folder.glob("*.png")):
                arr = np.array(PILImage.open(p).convert("RGB").resize((299, 299)))
                imgs.append(torch.tensor(arr).permute(2, 0, 1))
            return torch.stack(imgs) if imgs else None

        print("[4.3] Computing FID per condition …")
        for cond in conditions_eval:
            fid_fn = FrechetInceptionDistance(normalize=False).to(device)
            added_real = 0
            added_fake = 0
            for rec in records:
                pid     = rec["prompt_id"]
                pid_dir = img_root / str(pid)
                real_t  = load_uint8_tensor(pid_dir / "target")
                fake_t  = load_uint8_tensor(pid_dir / cond)
                if real_t is not None:
                    fid_fn.update(real_t.to(device), real=True)
                    added_real += len(real_t)
                if fake_t is not None:
                    fid_fn.update(fake_t.to(device), real=False)
                    added_fake += len(fake_t)
            if added_real >= 2 and added_fake >= 2:
                fid_score = float(fid_fn.compute())
                fid_per_cond[cond] = round(fid_score, 2)
                print(f"  FID [{cond}]: {fid_score:.2f}  (real={added_real}, fake={added_fake})")
            else:
                print(f"  FID [{cond}]: skipped (insufficient images)")
    except ImportError:
        print("[4.3] torchmetrics not installed — skipping FID. "
              "Run: pip install torchmetrics[image]")
    except Exception as e:
        print(f"[4.3] FID computation failed: {e}")

    # Wrap per-prompt results and FID together
    output = {"fid": fid_per_cond, "per_prompt": all_results}

    metrics_path = OUTPUT_DIR / f"metrics_{model_key}.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[4.3] [{model_key}] Metrics saved → {metrics_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 5: REPORT
# ══════════════════════════════════════════════════════════════════════════════

def run_report(model_key: str = "flux"):
    candidates   = json.loads((OUTPUT_DIR / "candidates.json").read_text())
    metrics_path = OUTPUT_DIR / f"metrics_{model_key}.json"
    if not metrics_path.exists():
        print(f"[4.3] Run --mode evaluate --model {model_key} first.")
        return

    with open(metrics_path) as f:
        raw = json.load(f)

    # Support both old format {pid: ...} and new format {fid: ..., per_prompt: {pid: ...}}
    if "per_prompt" in raw:
        all_results = raw["per_prompt"]
        fid_scores  = raw.get("fid", {})
    else:
        all_results = raw
        fid_scores  = {}

    conditions_eval = (["subject", "subj+all"]
                       + [f"subj+{c}" for c in candidates])

    # ── Table 2: Image-level metrics ──────────────────────────────────────────
    table2_rows = []
    for cond in conditions_eval:
        ssim_all, lpips_all = [], []
        for pid_data in all_results.values():
            if cond in pid_data:
                ssim_all.extend(pid_data[cond]["ssim"])
                lpips_all.extend(pid_data[cond]["lpips"])
        table2_rows.append({
            "Condition":  cond,
            "SSIM mean":  round(np.mean(ssim_all), 4) if ssim_all else None,
            "SSIM std":   round(np.std(ssim_all),  4) if ssim_all else None,
            "LPIPS mean": round(np.mean(lpips_all), 4) if lpips_all else None,
            "LPIPS std":  round(np.std(lpips_all),  4) if lpips_all else None,
            "FID":        fid_scores.get(cond),
        })
    df2 = pd.DataFrame(table2_rows)
    csv2 = OUTPUT_DIR / f"table2_image_metrics_{model_key}.csv"
    df2.to_csv(csv2, index=False)

    # ── Wilcoxon tests: Subj+c vs Subject (SSIM) ──────────────────────────────
    wilcoxon: dict[str, dict] = {}
    for cat in candidates:
        cond = f"subj+{cat}"
        base_ssim, cond_ssim = [], []
        for pid_data in all_results.values():
            if "subject" in pid_data and cond in pid_data:
                base_ssim.append(np.mean(pid_data["subject"]["ssim"]))
                cond_ssim.append(np.mean(pid_data[cond]["ssim"]))
        if len(base_ssim) >= 5:
            stat, p = stats.wilcoxon(cond_ssim, base_ssim, alternative="greater")
            wilcoxon[cat] = {"stat": float(stat), "p": float(p), "n": len(base_ssim)}
        else:
            wilcoxon[cat] = {"stat": None, "p": None, "n": len(base_ssim)}

    # ── Table 3: MSCS + t-test ────────────────────────────────────────────────
    table3_rows = []
    for cat in CATS:
        cond = f"subj+{cat}"
        mscs_with, mscs_without = [], []
        for pid_data in all_results.values():
            if cond in pid_data:
                mscs_with.append(pid_data[cond]["mscs"].get(cat))
            if "subject" in pid_data:
                mscs_without.append(pid_data["subject"]["mscs"].get(cat))

        mscs_with    = [v for v in mscs_with    if v is not None]
        mscs_without = [v for v in mscs_without if v is not None]

        if mscs_with and mscs_without and len(mscs_with) == len(mscs_without):
            diffs    = [w - wo for w, wo in zip(mscs_with, mscs_without)]
            mscs_val = float(np.mean(diffs))
            t_stat, t_p = stats.ttest_1samp(diffs, 0)
            t_p = float(t_p)
        else:
            mscs_val, t_stat, t_p = None, None, None

        w = wilcoxon.get(cat, {})
        is_steering = (
            cat in candidates
            and mscs_val is not None and mscs_val > 0
            and t_p is not None and t_p < 0.05
            and w.get("p") is not None and w["p"] < 0.05
        )
        table3_rows.append({
            "Category":   cat,
            "Candidate":  "✓" if cat in candidates else "✗",
            "MSCS":       round(mscs_val, 4) if mscs_val is not None else "N/A",
            "t-test p":   round(t_p, 4)      if t_p is not None      else "N/A",
            "Wilcoxon p": round(w["p"], 4)   if w.get("p") is not None else "N/A",
            "Steering":   "✓" if is_steering else "✗",
        })

    df3 = pd.DataFrame(table3_rows)
    csv3 = OUTPUT_DIR / f"table3_mscs_{model_key}.csv"
    df3.to_csv(csv3, index=False)

    # ── Spearman: frequency rank vs MSCS rank ─────────────────────────────────
    freq_df = pd.read_csv(OUTPUT_DIR / "table1_frequency.csv")
    freq_rank = {row["Category"]: i for i, (_, row) in
                 enumerate(freq_df.sort_values("f_p (%)", ascending=False).iterrows())}
    mscs_map  = {r["Category"]: r["MSCS"] for _, r in df3.iterrows()
                 if isinstance(r["MSCS"], float)}
    shared    = [c for c in mscs_map if c in freq_rank]
    if len(shared) >= 3:
        fr    = [freq_rank[c] for c in shared]
        mr    = pd.Series([mscs_map[c] for c in shared]).rank().tolist()
        rho, spear_p = stats.spearmanr(fr, mr)
    else:
        rho, spear_p = None, None

    steering_cats = [r["Category"] for _, r in df3.iterrows() if r["Steering"] == "✓"]

    # ── Print + write report ──────────────────────────────────────────────────
    lines = [
        "══════════════════════════════════════════════════════════════════════",
        "SECTION 4.3 — STEERING MODIFIER IDENTIFICATION RESULTS",
        "══════════════════════════════════════════════════════════════════════",
        "",
        "── Step 1: Candidate Categories (f_p > 20%) ─────────────────────────",
        f"  Candidates     : {candidates}",
        f"  Non-candidates : {[c for c in CATS if c not in candidates]}",
        "",
        "── Step 2: Image-level Validation (Wilcoxon, SSIM) ──────────────────",
        f"  {'Category':<12} {'p-value':>10}  {'sig':>5}",
        "  " + "─" * 32,
    ]
    for cat in candidates:
        w    = wilcoxon.get(cat, {})
        p_s  = f"{w['p']:.4f}" if w.get("p") is not None else "N/A"
        sig  = "✓" if w.get("p") is not None and w["p"] < 0.05 else "✗"
        lines.append(f"  {cat:<12} {p_s:>10}  {sig:>5}")

    lines += [
        "",
        "── Step 2: Semantic-level Validation (MSCS, t-test) ─────────────────",
        f"  {'Category':<12} {'MSCS':>8}  {'t-test p':>10}  {'sig':>5}",
        "  " + "─" * 42,
    ]
    for _, r in df3.iterrows():
        sig = "✓" if str(r["Steering"]) == "✓" else "✗"
        lines.append(f"  {r['Category']:<12} {str(r['MSCS']):>8}  "
                     f"{str(r['t-test p']):>10}  {sig:>5}")

    lines += [
        "",
        f"── Confirmed Steering Modifiers ─────────────────────────────────────",
        f"  {steering_cats}",
        "",
    ]
    if rho is not None:
        lines += [
            "── Frequency–Influence Spearman Correlation ─────────────────────────",
            f"  ρ = {rho:.4f},  p = {spear_p:.4f}",
            f"  {'✓ Significant' if spear_p < 0.05 else '✗ Not significant'}"
            f"  (expected ρ > 0.7)",
            "",
        ]
    lines += [
        "── Output Files ──────────────────────────────────────────────────────",
        f"  Table 1 → {OUTPUT_DIR}/table1_frequency.csv",
        f"  Table 2 → {csv2}",
        f"  Table 3 → {csv3}",
        "══════════════════════════════════════════════════════════════════════",
    ]

    report = "\n".join(lines)
    print(report)
    rpath = OUTPUT_DIR / "steering_report.txt"
    rpath.write_text(report)
    print(f"\n[4.3] Report → {rpath}")


# ══════════════════════════════════════════════════════════════════════════════
# TARGETED MODES: Repeating and Magic (n=10 each, independent sampling)
# ══════════════════════════════════════════════════════════════════════════════

def run_sample_targeted(cat: str):
    """
    Sample N_TARGETED_PROMPTS prompts that contain ≥1 modifier of `cat`.
    No requirement for other categories — avoids selection bias for low-f_p cats.
    Output: outputs/4_3/targeted_{cat.lower()}_prompts.jsonl
    """
    import random
    random.seed(SEED)

    if cat not in CATS:
        print(f"[targeted] Unknown category: {cat}. Choose from {CATS}")
        return

    print(f"[targeted] Sampling {N_TARGETED_PROMPTS} prompts for category: {cat}")
    records = load_records()

    EXCLUDE_IDS = {6921, 69491}
    MIN_CONTENT_WORDS = 3
    _LEAK_RE = re.compile(
        r'\bartstation\b|\bdeviantart\b|\bpixiv\b|\bcgsociety\b|\bhearthstone\b'
        r'|\bart by\b|\bby [A-Z][a-z]'
        r'|\balphonse mucha\b|\bgreg rutkowski\b|\bartgerm\b|\bwlop\b'
        r'|\boctane render\b|\bunreal engine\b|\bbeeple\b',
        re.IGNORECASE,
    )

    # For Repeating: require at least one span with a truly repeated word
    # (rules out spans like "symmetry!!" that are only repeated punctuation)
    _WORD_REPEAT_RE = re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE)

    def eligible(rec):
        by_cat = get_cat_spans(rec)
        if len(by_cat[cat]) == 0:
            return False
        if cat == "Repeating":
            if not any(_WORD_REPEAT_RE.search(m.get("span", ""))
                       for m in by_cat[cat]):
                return False
        if rec.get("prompt_id") in EXCLUDE_IDS:
            return False
        subject = extract_subject(rec["prompt"], rec.get("modifiers", []))
        if re.search(r'[\x80-\xff]|ï¿½', subject):
            return False
        if _LEAK_RE.search(subject):
            return False
        return content_word_count(subject) >= MIN_CONTENT_WORDS

    pool = [r for r in records if eligible(r)]
    print(f"[targeted] Eligible for {cat}: {len(pool):,}")

    random.shuffle(pool)

    # For Repeating: stratify so that at most half come from "very" repeats,
    # filling remaining slots with other repeated words first.
    if cat == "Repeating":
        def primary_repeated_word(rec):
            for m in get_cat_spans(rec)[cat]:
                hit = _WORD_REPEAT_RE.search(m.get("span", ""))
                if hit:
                    return hit.group(1).lower()
            return "very"

        non_very = [r for r in pool if primary_repeated_word(r) != "very"]
        very     = [r for r in pool if primary_repeated_word(r) == "very"]
        max_very = N_TARGETED_PROMPTS // 2          # at most 5 "very" prompts
        selected = non_very[:N_TARGETED_PROMPTS] + very[:max_very]
        # de-duplicate by prompt_id, keep order
        seen = set()
        deduped = []
        for r in selected:
            if r["prompt_id"] not in seen:
                seen.add(r["prompt_id"])
                deduped.append(r)
        selected = deduped[:N_TARGETED_PROMPTS]
        print(f"  non-very: {len([r for r in selected if primary_repeated_word(r) != 'very'])}, "
              f"very: {len([r for r in selected if primary_repeated_word(r) == 'very'])}")
    else:
        selected = pool[:N_TARGETED_PROMPTS]

    if len(selected) < N_TARGETED_PROMPTS:
        print(f"  WARNING: only {len(selected)} prompts available for {cat}")

    output_records = []
    for rec in selected:
        subject = extract_subject(rec["prompt"], rec.get("modifiers", []))
        by_cat  = get_cat_spans(rec)
        # For Repeating: use only spans with truly repeated words
        spans = by_cat[cat]
        if cat == "Repeating":
            spans = [m for m in spans if _WORD_REPEAT_RE.search(m.get("span", ""))]
        spans_text = ", ".join(m["span"] for m in spans)
        subprompts = {
            "subject": subject,
            f"subj+{cat}": (subject + ", " + spans_text).strip(", "),
        }
        output_records.append({
            "prompt_id":  rec["prompt_id"],
            "prompt":     rec["prompt"],
            "category":   cat,
            "subprompts": subprompts,
            "modifiers":  rec.get("modifiers", []),
        })

    out_path = OUTPUT_DIR / f"targeted_{cat.lower()}_prompts.jsonl"
    with open(out_path, "w") as f:
        for rec in output_records:
            f.write(json.dumps(rec) + "\n")

    # Human-readable preview
    txt_path = OUTPUT_DIR / f"targeted_{cat.lower()}_prompts.txt"
    with open(txt_path, "w") as f:
        f.write(f"Targeted {cat} prompts (n={len(output_records)})\n")
        f.write("=" * 70 + "\n\n")
        for i, rec in enumerate(output_records, 1):
            f.write(f"── [{i:02d}] id={rec['prompt_id']}\n")
            f.write(f"ORIGINAL : {rec['prompt']}\n")
            for cond, text in rec["subprompts"].items():
                f.write(f"  {cond:<16}: {text[:100]}{'…' if len(text) > 100 else ''}\n")
            f.write("\n")

    print(f"[targeted] Saved → {out_path}")
    print(f"[targeted] Preview → {txt_path}")


def run_generate_targeted(cat: str, model_key: str = "flux"):
    """Generate subj + subj+{cat} + target images for targeted prompts."""
    import time
    import torch

    in_path = OUTPUT_DIR / f"targeted_{cat.lower()}_prompts.jsonl"
    if not in_path.exists():
        print(f"[targeted] Run --mode sample_targeted --cat {cat} first.")
        return

    records = []
    with open(in_path) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"[targeted] Loading {model_key} ({MODELS[model_key]}) …")
    pipe = _load_pipeline(model_key)
    generator = torch.Generator("cuda").manual_seed(SEED)

    img_root = OUTPUT_DIR / "images" / model_key / f"targeted_{cat.lower()}"
    img_root.mkdir(parents=True, exist_ok=True)

    conditions = ["subject", f"subj+{cat}", "target"]
    n_per_cond = {"target": N_TARGET_IMAGES}

    tasks: list[tuple[str, Path]] = []
    for rec in records:
        pid = rec["prompt_id"]
        subs = rec["subprompts"]
        (img_root / str(pid)).mkdir(exist_ok=True)
        for cond in conditions:
            n = n_per_cond.get(cond, N_IMAGES_PER_SUBPROMPT)
            cond_dir = img_root / str(pid) / cond
            cond_dir.mkdir(exist_ok=True)
            prompt_text = rec["prompt"] if cond == "target" else subs.get(cond, "")
            if not prompt_text:
                continue
            for idx in range(n):
                out = cond_dir / f"{idx:02d}.png"
                if not out.exists():
                    tasks.append((prompt_text, out))

    total = len(tasks)
    print(f"[targeted] {total} images to generate")

    BATCH = 16 if model_key == "flux" else 12
    pipe_kwargs = (
        dict(num_inference_steps=4,  guidance_scale=0.0)
        if model_key == "flux" else
        dict(num_inference_steps=30, guidance_scale=7.5)
    )

    done = 0
    t0   = time.time()
    for i in range(0, total, BATCH):
        chunk     = tasks[i : i + BATCH]
        images    = pipe(
            prompt=[t[0] for t in chunk],
            num_images_per_prompt=1,
            generator=generator,
            **pipe_kwargs,
        ).images
        for img, path in zip(images, [t[1] for t in chunk]):
            img.save(path)
        done += len(chunk)
        elapsed = time.time() - t0
        eta     = elapsed / done * (total - done) if done else 0
        print(f"  [{done}/{total}]  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m",
              flush=True)

    print(f"\n[targeted] [{model_key}] {cat} generation complete → {img_root}")


def run_evaluate_targeted(cat: str, model_key: str = "flux"):
    """Compute SSIM / LPIPS / MSCS for targeted prompts (subj vs subj+{cat})."""
    import torch
    import clip
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim_fn
    import lpips as lpips_lib

    in_path  = OUTPUT_DIR / f"targeted_{cat.lower()}_prompts.jsonl"
    img_root = OUTPUT_DIR / "images" / model_key / f"targeted_{cat.lower()}"

    if not in_path.exists():
        print(f"[targeted] Missing {in_path}")
        return

    records = []
    with open(in_path) as f:
        for line in f:
            records.append(json.loads(line))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)

    # Text anchor for this specific category
    all_recs = load_records()
    counter: Counter = Counter()
    for rec in all_recs:
        for m in rec.get("modifiers", []):
            if m.get("category") == cat:
                counter[m["span"].lower().strip()] += 1
    terms = [t for t, _ in counter.most_common(3)] or [cat.lower()]
    tokens = clip.tokenize(terms, truncate=True).to(device)
    with torch.no_grad():
        e = clip_model.encode_text(tokens).float()
        e = e / e.norm(dim=-1, keepdim=True)
        anchor = e.mean(dim=0)
        text_anchor = anchor / anchor.norm()
    print(f"[targeted] T_{cat}: {terms}")

    def load_np(folder):
        return [np.array(Image.open(p).convert("RGB").resize((512, 512)))
                for p in sorted(folder.glob("*.png"))]

    def load_tensor(folder):
        imgs = [torch.tensor(np.array(Image.open(p).convert("RGB").resize((512, 512))))
                .permute(2, 0, 1).float() / 127.5 - 1.0
                for p in sorted(folder.glob("*.png"))]
        return torch.stack(imgs).to(device) if imgs else None

    def load_clip(folder):
        imgs = [clip_preprocess(Image.open(p).convert("RGB"))
                for p in sorted(folder.glob("*.png"))]
        return torch.stack(imgs).to(device) if imgs else None

    def img_embed(clip_t):
        with torch.no_grad():
            e = clip_model.encode_image(clip_t).float()
            e = e / e.norm(dim=-1, keepdim=True)
        return e.mean(dim=0)

    conditions_eval = ["subject", f"subj+{cat}"]
    all_results: dict = {}

    for rec in records:
        pid     = rec["prompt_id"]
        pid_dir = img_root / str(pid)
        print(f"[targeted] Evaluating {cat} prompt {pid} …")

        target_dir = pid_dir / "target"
        if not target_dir.exists() or not list(target_dir.glob("*.png")):
            print(f"  WARNING: no target images for {pid}, skipping")
            continue

        target_np   = load_np(target_dir)
        target_ref  = target_np[0]
        target_clip = load_clip(target_dir)
        target_t    = load_tensor(target_dir)

        result: dict = {}
        for cond in conditions_eval:
            cond_dir = pid_dir / cond
            if not cond_dir.exists():
                continue
            cond_np   = load_np(cond_dir)
            ssim_vals = [float(ssim_fn(target_ref, img, channel_axis=2, data_range=255))
                         for img in cond_np]
            cond_t    = load_tensor(cond_dir)
            lpips_vals = []
            if cond_t is not None and target_t is not None:
                t_ref = target_t[:1].expand(len(cond_t), -1, -1, -1)
                with torch.no_grad():
                    lp = lpips_fn(cond_t, t_ref).squeeze().cpu().numpy()
                lpips_vals = lp.tolist() if lp.ndim > 0 else [float(lp)]
            cond_clip = load_clip(cond_dir)
            mscs_val  = None
            if cond_clip is not None:
                cond_embed = img_embed(cond_clip)
                mscs_val   = float(torch.dot(cond_embed, text_anchor).item())
            result[cond] = {"ssim": ssim_vals, "lpips": lpips_vals, "mscs": mscs_val}

        all_results[str(pid)] = result

    metrics_path = OUTPUT_DIR / f"metrics_targeted_{cat.lower()}_{model_key}.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[targeted] [{model_key}] Metrics saved → {metrics_path}")


def run_report_targeted():
    """
    Aggregate targeted results for Repeating and Magic across both models.
    Prints / saves Table 5 (dual-metric results).
    """
    rows = []
    for cat in TARGETED_CATS:
        for model_key in MODELS:
            metrics_path = OUTPUT_DIR / f"metrics_targeted_{cat.lower()}_{model_key}.json"
            if not metrics_path.exists():
                print(f"[targeted] Missing {metrics_path} — run evaluate_targeted first")
                continue
            with open(metrics_path) as f:
                all_results = json.load(f)

            cond_c    = f"subj+{cat}"
            base_ssim, cond_ssim = [], []
            base_mscs, cond_mscs = [], []
            base_lpips, cond_lpips = [], []

            for pid_data in all_results.values():
                if "subject" in pid_data and cond_c in pid_data:
                    base_ssim.append(np.mean(pid_data["subject"]["ssim"]))
                    cond_ssim.append(np.mean(pid_data[cond_c]["ssim"]))
                    base_lpips.append(np.mean(pid_data["subject"]["lpips"]) if pid_data["subject"]["lpips"] else np.nan)
                    cond_lpips.append(np.mean(pid_data[cond_c]["lpips"])  if pid_data[cond_c]["lpips"] else np.nan)
                    if pid_data["subject"]["mscs"] is not None and pid_data[cond_c]["mscs"] is not None:
                        base_mscs.append(pid_data["subject"]["mscs"])
                        cond_mscs.append(pid_data[cond_c]["mscs"])

            n = len(base_ssim)
            if n < 5:
                print(f"  WARNING: only {n} paired samples for {cat}/{model_key}, skipping stats")
                continue

            # Wilcoxon (SSIM)
            stat_w, p_w = stats.wilcoxon(cond_ssim, base_ssim, alternative="greater")

            # MSCS difference t-test
            if cond_mscs:
                diffs    = [c - b for c, b in zip(cond_mscs, base_mscs)]
                mscs_delta = float(np.mean(diffs))
                _, p_t   = stats.ttest_1samp(diffs, 0)
                p_t      = float(p_t)
            else:
                mscs_delta, p_t = None, None

            is_steering = (
                mscs_delta is not None and mscs_delta > 0
                and p_t is not None and p_t < 0.05
                and p_w < 0.05
            )

            rows.append({
                "Category":     cat,
                "Model":        model_key.upper(),
                "N":            n,
                "SSIM (subj+c)": round(float(np.mean(cond_ssim)), 4),
                "SSIM (subj)":  round(float(np.mean(base_ssim)), 4),
                "LPIPS (subj+c)": round(float(np.nanmean(cond_lpips)), 4) if cond_lpips else "N/A",
                "LPIPS (subj)": round(float(np.nanmean(base_lpips)), 4) if base_lpips else "N/A",
                "MSCS Δ":       round(mscs_delta, 4) if mscs_delta is not None else "N/A",
                "Wilcoxon p":   round(float(p_w), 4),
                "t-test p":     round(p_t, 4) if p_t is not None else "N/A",
                "Steering":     "✓" if is_steering else "✗",
            })

    if not rows:
        print("[targeted] No results to report.")
        return

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "table5_targeted.csv"
    df.to_csv(csv_path, index=False)

    print("\n── Table 5: Targeted Results (Repeating & Magic) ────────────────────")
    print(df.to_string(index=False))
    print(f"\n[targeted] Table 5 → {csv_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["freq", "sample", "generate", "evaluate", "report",
                                 "sample_targeted", "generate_targeted",
                                 "evaluate_targeted", "report_targeted"])
    parser.add_argument("--model", default="flux", choices=list(MODELS.keys()),
                        help="T2I model for generate/evaluate/report (default: flux)")
    parser.add_argument("--cat", default=None,
                        help="Category for targeted modes (e.g. Repeating, Magic)")
    args = parser.parse_args()

    if args.mode == "freq":
        run_freq()
    elif args.mode == "sample":
        run_sample()
    elif args.mode == "generate":
        run_generate(args.model)
    elif args.mode == "evaluate":
        run_evaluate(args.model)
    elif args.mode == "report":
        run_report(args.model)
    elif args.mode == "sample_targeted":
        cats = [args.cat] if args.cat else TARGETED_CATS
        for cat in cats:
            run_sample_targeted(cat)
    elif args.mode == "generate_targeted":
        cats = [args.cat] if args.cat else TARGETED_CATS
        for cat in cats:
            run_generate_targeted(cat, args.model)
    elif args.mode == "evaluate_targeted":
        cats = [args.cat] if args.cat else TARGETED_CATS
        for cat in cats:
            run_evaluate_targeted(cat, args.model)
    elif args.mode == "report_targeted":
        run_report_targeted()


if __name__ == "__main__":
    main()

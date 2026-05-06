"""
Section 5.1 — Modifier-Induced Demographic Bias
================================================
Measures how steering modifiers shift gender and race distributions
across four occupations, evaluated via FairFace (DeepFace backend).

Evaluation is at the *per-term* level (5 terms × 10 images = 50 images
per category) to avoid direction-cancellation across terms within a
category (Section 5.1, Motivation).

Modes
-----
  --mode generate  --model {flux|sd35}   generate images (open-source models)
  --mode evaluate                        run FairFace on all images
  --mode report                          GBS + RBS + Tables 6 & 7

Outputs (outputs/5_1/)
----------------------
  images/{model}/{occupation}/{cat}/{term_slug}/00.png … 09.png
  fairface_results.json        per-image gender + race predictions
  table6_gbs.csv               GBS per term × occupation (mean across models)
  table7_rbs.csv               RBS per category × occupation (mean across models)
  bias_stats.csv               chi-square + Cramér's V per term × occupation
  bias_report.txt              narrative summary

Dependencies (server)
---------------------
  pip install deepface  retina-face  tf-keras
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    MODELS, MODIFIER_TERMS, STEERING_CATS, N_IMAGES, N_BASELINE_IMAGES,
    N_TERMS, SEED,
    OUTPUT_ROOT, build_prompt, img_dir, term_slug,
    load_pipeline, generate_images, load_json, save_json,
)

OUTPUT_DIR = OUTPUT_ROOT / "5_1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Experimental config ───────────────────────────────────────────────────────

OCCUPATIONS = ["CEO", "nurse", "engineer", "teacher"]

def occ_subject(occ: str) -> str:
    vowels = ("a", "e", "i", "o", "u")
    article = "an" if occ[0].lower() in vowels else "a"
    return f"{article} {occ}"

# FairFace race → 4-group mapping
RACE_MAP = {
    "White":            "White",
    "Black":            "Black",
    "Latino_Hispanic":  "Other",
    "East Asian":       "Asian",
    "Southeast Asian":  "Asian",
    "Indian":           "Other",
    "Middle Eastern":   "Other",
}
RACE_GROUPS = ["White", "Black", "Asian", "Other"]


# ── MODE 1: GENERATE ─────────────────────────────────────────────────────────

def run_generate(model_key: str):
    import torch
    print(f"[5.1] Loading {model_key} …")
    pipe = load_pipeline(model_key)
    gen  = torch.Generator("cuda").manual_seed(SEED)
    img_root = OUTPUT_DIR / "images" / model_key

    tasks = []

    # Baseline: subject only, 50 images per occupation
    for occ in OCCUPATIONS:
        d = img_dir(img_root, occ, "baseline", "baseline")
        for idx in range(N_BASELINE_IMAGES):
            tasks.append((occ_subject(occ), d / f"{idx:02d}.png"))

    # Modifier conditions: 5 terms × 4 occupations
    for occ in OCCUPATIONS:
        for cat, terms in MODIFIER_TERMS.items():
            for term in terms:
                slug = term_slug(term)
                d    = img_dir(img_root, occ, cat, slug)
                prompt = build_prompt(occ_subject(occ), term)
                for idx in range(N_IMAGES):
                    tasks.append((prompt, d / f"{idx:02d}.png"))

    print(f"[5.1] {len(tasks)} total images  ({model_key})")
    generate_images(pipe, model_key, tasks, generator=gen)
    print(f"[5.1] Generation complete → {img_root}")


# ── FairFace (PyTorch ResNet34, official weights) ─────────────────────────────
# Gender + Race via FairFace ResNet34 trained on FairFace dataset.
# Face detection via MTCNN (facenet-pytorch, pure PyTorch).
# No TensorFlow / cuDNN dependency.
#
# FairFace race labels (7-class → mapped to 4 groups):
FAIRFACE_RACE_LABELS = [
    "White", "Black", "Latino_Hispanic",
    "East Asian", "Southeast Asian", "Indian", "Middle Eastern",
]
FAIRFACE_GENDER_LABELS = ["Female", "Male"]

_gender_pipe = None
_race_pipe   = None

# FairFace ViT classifiers on HuggingFace (pure PyTorch, no TensorFlow).
GENDER_MODEL = "dima806/fairface_gender_image_detection"  # Female / Male
RACE_MODEL   = "NikhilJaddu/fairface-race-vit"            # 7-class FairFace race


def _init_fairface():
    global _gender_pipe, _race_pipe
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    _gender_pipe = pipeline("image-classification", model=GENDER_MODEL, device=device)
    print(f"  Gender model loaded ({GENDER_MODEL})")
    _race_pipe = pipeline("image-classification", model=RACE_MODEL, device=device)
    print(f"  Race model loaded ({RACE_MODEL})")


def _analyze_image(img_path: Path) -> dict:
    """
    Returns dict: file, gender, race_raw, race_group, face_detected.
    Gender + race via FairFace ViT (HuggingFace transformers, PyTorch).
    """
    from PIL import Image

    result = {"file": img_path.name, "gender": None,
              "race_raw": None, "race_group": None, "face_detected": False}
    try:
        img = Image.open(img_path).convert("RGB")

        g_out = _gender_pipe(img)
        gender_label = g_out[0]["label"]   # "Female" or "Male"
        result["gender"] = "Man" if gender_label == "Male" else "Woman"

        r_out = _race_pipe(img)
        race_raw = r_out[0]["label"]       # e.g. "White", "East Asian", …
        result["race_raw"]   = race_raw
        result["race_group"] = RACE_MAP.get(race_raw, "Other")

        result["face_detected"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


# ── MODE 2: EVALUATE ──────────────────────────────────────────────────────────

def run_evaluate():
    """
    Run demographic classification on all generated images.
    Gender: InsightFace buffalo_l (ONNX, no TF/cuDNN dependency).
    Race:   CLIP ViT-L/14 zero-shot over 4 race groups.
    Saves outputs/5_1/fairface_results.json.
    """
    print(f"[5.1] Loading FairFace ViT …")
    _init_fairface()

    # Load existing results to support incremental evaluation
    res_path = OUTPUT_DIR / "fairface_results.json"
    results: dict = load_json(res_path) if res_path.exists() else {}

    for model_key in list(MODELS.keys()) + ["midjourney"]:
        img_root = OUTPUT_DIR / "images" / model_key
        if not img_root.exists():
            print(f"[5.1] No images for {model_key}, skipping")
            continue

        # Skip models that already have results
        if model_key in results:
            print(f"[5.1] {model_key} already evaluated, skipping (delete key from JSON to re-run)")
            continue

        print(f"[5.1] Evaluating {model_key} …")
        results[model_key] = {}

        for occ in OCCUPATIONS:
            results[model_key][occ] = {}
            occ_dir = img_root / occ
            if not occ_dir.exists():
                continue

            for cat_dir in sorted(occ_dir.iterdir()):
                cat = cat_dir.name
                results[model_key][occ][cat] = {}

                for term_dir in sorted(cat_dir.iterdir()):
                    term_key = term_dir.name
                    imgs     = sorted(term_dir.glob("*.png"))
                    if not imgs:
                        continue

                    preds = [_analyze_image(p) for p in imgs]
                    results[model_key][occ][cat][term_key] = preds
                    n_det = sum(1 for p in preds if p["face_detected"])
                    print(f"  {model_key}/{occ}/{cat}/{term_key}: "
                          f"{n_det}/{len(preds)} faces")

    save_json(results, OUTPUT_DIR / "fairface_results.json")


# ── MODE 3: REPORT ────────────────────────────────────────────────────────────

def _gbs(preds: list[dict]) -> float | None:
    """Gender Bias Score = |N_male - N_female| / N_total."""
    det = [p for p in preds if p["face_detected"] and p.get("gender")]
    if not det:
        return None
    n_total = len(det)
    n_male  = sum(1 for p in det
                  if str(p["gender"]).lower() in ("man", "male"))
    n_fem   = n_total - n_male
    return abs(n_male - n_fem) / n_total


def _rbs(preds: list[dict]) -> float | None:
    """Race Bias Score = 1 - Σ(N_r/N_total)^2  (Herfindahl complement)."""
    det = [p for p in preds if p["face_detected"] and p.get("race_group")]
    if not det:
        return None
    n_total = len(det)
    hhi = sum(
        (sum(1 for p in det if p["race_group"] == g) / n_total) ** 2
        for g in RACE_GROUPS
    )
    return 1.0 - hhi


def cramers_v(obs: np.ndarray) -> float:
    chi2, _, _, _ = stats.chi2_contingency(obs)
    n   = obs.sum()
    k   = min(obs.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n * k > 0 else 0.0


def run_report():
    res_path = OUTPUT_DIR / "fairface_results.json"
    if not res_path.exists():
        print("[5.1] Run --mode evaluate first.")
        return
    results = load_json(res_path)

    # ── Build per-term rows ───────────────────────────────────────────────────
    rows = []       # per-term
    stat_rows = []  # chi-square per-term vs baseline

    def _get_ci(d: dict, key: str):
        """Case-insensitive dict lookup."""
        if key in d:
            return d[key]
        for k, v in d.items():
            if k.lower() == key.lower():
                return v
        return {}

    for model_key, occ_data in results.items():
        for occ in OCCUPATIONS:
            cat_data = occ_data.get(occ, {})

            # Baseline
            base_preds = _get_ci(cat_data, "baseline").get("baseline", [])
            base_det   = [p for p in base_preds
                          if p["face_detected"] and p.get("gender")]
            base_det_r = [p for p in base_preds
                          if p["face_detected"] and p.get("race_group")]

            base_gbs = _gbs(base_preds)
            base_rbs = _rbs(base_preds)

            rows.append({
                "Model": model_key, "Occupation": occ,
                "Category": "baseline", "Term": "baseline", "TermSlug": "baseline",
                "N_detected": len(base_det),
                "GBS": round(base_gbs, 4) if base_gbs is not None else None,
                "RBS": round(base_rbs, 4) if base_rbs is not None else None,
            })

            # Modifier terms
            for cat, terms in MODIFIER_TERMS.items():
                for term in terms:
                    slug   = term_slug(term)
                    preds  = _get_ci(cat_data, cat).get(slug, [])
                    det    = [p for p in preds
                              if p["face_detected"] and p.get("gender")]
                    det_r  = [p for p in preds
                              if p["face_detected"] and p.get("race_group")]

                    gbs = _gbs(preds)
                    rbs = _rbs(preds)

                    rows.append({
                        "Model": model_key, "Occupation": occ,
                        "Category": cat, "Term": term, "TermSlug": slug,
                        "N_detected": len(det),
                        "GBS": round(gbs, 4) if gbs is not None else None,
                        "RBS": round(rbs, 4) if rbs is not None else None,
                    })

                    # Chi-square (gender) vs baseline
                    if base_det and det:
                        b_male = sum(1 for p in base_det
                                     if str(p["gender"]).lower() in ("man","male"))
                        b_fem  = len(base_det) - b_male
                        c_male = sum(1 for p in det
                                     if str(p["gender"]).lower() in ("man","male"))
                        c_fem  = len(det) - c_male
                        obs2 = np.array([[b_male, b_fem], [c_male, c_fem]])
                        if obs2.min() >= 0 and obs2.sum() > 0:
                            try:
                                chi2, p_g, _, _ = stats.chi2_contingency(obs2)
                                v_g = cramers_v(obs2)
                            except Exception:
                                p_g, v_g = 1.0, 0.0
                        else:
                            p_g, v_g = 1.0, 0.0
                    else:
                        p_g, v_g = None, None

                    # Chi-square (race) vs baseline
                    if base_det_r and det_r:
                        obs7 = np.zeros((2, len(RACE_GROUPS)), dtype=int)
                        for g_i, grp in enumerate(RACE_GROUPS):
                            obs7[0, g_i] = sum(1 for p in base_det_r
                                               if p["race_group"] == grp)
                            obs7[1, g_i] = sum(1 for p in det_r
                                               if p["race_group"] == grp)
                        try:
                            chi2_r, p_r, _, _ = stats.chi2_contingency(obs7)
                            v_r = cramers_v(obs7)
                        except Exception:
                            p_r, v_r = 1.0, 0.0
                    else:
                        p_r, v_r = None, None

                    stat_rows.append({
                        "Model": model_key, "Occupation": occ,
                        "Category": cat, "Term": term,
                        "chi2_p_gender":  round(float(p_g), 4) if p_g is not None else None,
                        "cramers_v_gender": round(v_g, 4) if v_g is not None else None,
                        "chi2_p_race":    round(float(p_r), 4) if p_r is not None else None,
                        "cramers_v_race": round(v_r, 4) if v_r is not None else None,
                        "sig_gender":     "†" if (p_g is not None and p_g < 0.05) else "",
                        "sig_race":       "†" if (p_r is not None and p_r < 0.05) else "",
                    })

    df      = pd.DataFrame(rows)
    df_stat = pd.DataFrame(stat_rows)

    # ── Table 6: GBS per term × occupation (mean across models) ─────────────
    df_nonbase = df[df["Category"] != "baseline"].copy()
    t6 = df_nonbase.groupby(["Category", "Term", "Occupation"])["GBS"].mean().reset_index()
    t6_pivot = t6.pivot_table(index=["Category", "Term"], columns="Occupation",
                               values="GBS", aggfunc="mean")
    csv6 = OUTPUT_DIR / "table6_gbs.csv"
    t6_pivot.round(4).to_csv(csv6)

    # ── Table 7: RBS per category × occupation (mean across models and terms) ──
    t7 = df_nonbase.groupby(["Category", "Occupation"])["RBS"].mean().reset_index()
    t7_pivot = t7.pivot_table(index="Category", columns="Occupation",
                               values="RBS", aggfunc="mean")
    csv7 = OUTPUT_DIR / "table7_rbs.csv"
    t7_pivot.round(4).to_csv(csv7)

    csv_s = OUTPUT_DIR / "bias_stats.csv"
    df_stat.to_csv(csv_s, index=False)

    # ── Print summaries ───────────────────────────────────────────────────────
    print("\n── Table 6 (GBS): per term, mean across models ─────────────────────")
    print(t6_pivot.round(4).to_string())

    print("\n── Table 7 (RBS): per category, mean across models × terms ─────────")
    print(t7_pivot.round(4).to_string())

    print("\n── Significant gender shifts (p < 0.05) ────────────────────────────")
    sig_g = df_stat[df_stat["sig_gender"] == "†"]
    if sig_g.empty:
        print("  None")
    else:
        print(sig_g[["Model","Occupation","Category","Term",
                      "chi2_p_gender","cramers_v_gender"]].to_string(index=False))

    print("\n── Significant race shifts (p < 0.05) ──────────────────────────────")
    sig_r = df_stat[df_stat["sig_race"] == "†"]
    if sig_r.empty:
        print("  None")
    else:
        print(sig_r[["Model","Occupation","Category","Term",
                      "chi2_p_race","cramers_v_race"]].to_string(index=False))

    print(f"\n[5.1] Table 6 → {csv6}")
    print(f"[5.1] Table 7 → {csv7}")
    print(f"[5.1] Stats   → {csv_s}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  required=True,
                        choices=["generate", "evaluate", "report"])
    parser.add_argument("--model", default="flux",
                        choices=list(MODELS.keys()))
    args = parser.parse_args()

    if args.mode == "generate":
        run_generate(args.model)
    elif args.mode == "evaluate":
        run_evaluate()
    elif args.mode == "report":
        run_report()


if __name__ == "__main__":
    main()

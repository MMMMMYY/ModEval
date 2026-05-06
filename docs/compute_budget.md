# Compute Budget

This document estimates the compute required to reproduce each stage of the
ModEval pipeline. It is intended for the NeurIPS 2026 reproducibility
checklist and to help reviewers gauge feasibility.

All wall-time numbers are reported on a single **NVIDIA H100 80 GB** unless
otherwise noted; the same workloads run on an A100 40 GB will take roughly
1.7× longer.

---

## RQ1 — Steering Modifier Identification

| Stage | Script | Wall time | GPU memory | Disk I/O |
|-------|--------|----------:|-----------:|---------:|
| §3.1 Taxonomy harmonization (SBERT + clustering) | `section_4_1_harmonization.py` | ~ 30 s | ~ 1 GB | < 1 MB |
| §3.2 PMD construction (LLM annotation of 73 718 prompts via Llama-3.1-8B-Instruct) | `section_4_2_classification_vllm.py` | ~ 6 h (vLLM) | ~ 24 GB | ~ 70 MB output |
| §3.2 Pre-annotation gold validation (κ + per-category P/R/F1) | `compute_kappa.py` | < 5 s | CPU only | < 1 MB |
| §3.3 Image generation (25 prompts × 9 conditions × 4 samples × 2 models) | `section_4_3_steering.py` (gen mode) | ~ 5 h | ~ 24 GB | ~ 8 GB |
| §3.3 Targeted independent sampling (Repeating + Magic, 10 prompts × 2 conditions × 4 samples × 2 models) | `run_4_3_targeted.sh` | ~ 1.5 h | ~ 24 GB | ~ 1.5 GB |
| §3.3 Image-level + semantic-level metric computation (SSIM, LPIPS, MSCS) | `section_4_3_steering.py` (eval mode) | ~ 30 min | ~ 8 GB | < 1 MB |
| **RQ1 total**                                                                     |                                | **≈ 13 h** | **≤ 24 GB** | **≈ 10 GB** |

---

## RQ2 — Downstream Social-Risk Evaluation

For each of the three risk dimensions, generation and classifier inference are
the two cost drivers. All three commercial / open-source T2I models generate
at a budget of **5 representative spans × 10 images per span × 4 occupations
(or 2 subjects, or 2 NSFW types) × 3 models** per category; see paper §4.1.

### 4.2 Demographic Bias

| Stage | Script | Wall time | Notes |
|-------|--------|----------:|-------|
| Generation: 6 categories × 4 occupations × 5 spans × 10 images × 3 models = 3 600 images | `section_5_1_bias.py` (gen mode) | ~ 8 h | open-source (FLUX-dev, SD-3.5-Large) on GPU; Midjourney via web UI |
| Classification: FairFace gender ViT + race ViT on 3 600 images | `section_5_1_bias.py` (eval mode) | ~ 30 min | ~ 6 GB GPU |
| χ² + Cramér's V aggregation, Table 2 build | `figures_repro/5_1_bias/build_table6.py` | < 5 s | CPU |
| Figure 2/3 (per-occupation gender / race bars) | `figures_repro/5_1_bias/plot_figure*.py` | < 5 s | CPU |

### 4.3 Deepfake Detectability

| Stage | Script | Wall time | Notes |
|-------|--------|----------:|-------|
| Generation: 6 categories × 2 subjects × 5 spans × 10 images × 3 models = 1 800 images | `section_5_2_misuse.py` (gen mode) | ~ 4 h | same as above |
| Classification: prithivMLmods/Deep-Fake-Detector-v2-Model ViT | `section_5_2_misuse.py` (eval mode) | ~ 15 min | ~ 4 GB GPU |
| Fisher exact + DDR aggregation, Table 3 build | derived in `figures_repro/5_2_misuse/` | < 5 s | CPU |

### 4.4 Safety (NSFW / gore)

| Stage | Script | Wall time | Notes |
|-------|--------|----------:|-------|
| Generation: 6 categories × 2 NSFW subjects × 5 spans × 10 images × 3 models = 1 800 images | `section_5_3_safety.py` (gen mode) | ~ 4 h | same as above |
| NudeNet sexually-explicit classification | `section_5_3_safety.py` (eval mode) | ~ 5 min | CPU OK |
| Q16 gore classification | `run_q16_standalone.py` | ~ 10 min | ~ 4 GB GPU |
| Fisher exact + Δrate aggregation, Table 4 build | derived in `figures_repro/5_3_safety/` | < 5 s | CPU |

### 4.5 Cross-Dimension Synthesis (Figure 4)

| Stage | Script | Wall time |
|-------|--------|----------:|
| Live aggregation from 3 upstream CSVs + plot | `figures_repro/5_4_cross_risk/plot_figure10.py` | < 5 s |

**RQ2 total (excluding Midjourney generation): ≈ 17 h on a single H100 + ≈ 12 GB disk.**

---

## Summary

| Pipeline stage | Approx. GPU·h | Disk |
|----------------|--------------:|-----:|
| RQ1 full pipeline | ~ 13 GPU·h    | ~ 10 GB |
| RQ2 full pipeline | ~ 17 GPU·h    | ~ 12 GB |
| **Re-render figures only (frozen CSVs)** | **≈ 0.01 CPU·h** | **0 GB** |

Reviewers can re-render every figure and table in the paper with no GPU at all
(`bash run_all_figures.sh` from the repository root, < 30 s).

# Reproducibility Notes

This document explains what is and is not deterministic in the ModEval
pipeline, and how to reproduce numbers in the paper as faithfully as possible.

---

## What is fully deterministic

The following stages produce identical outputs across re-runs given the
shipped frozen data:

- **All figures and tables in the paper.** Running `bash run_all_figures.sh`
  re-renders every figure and table from frozen CSVs / JSONs under
  `exp_rq1/outputs/` and `exp_rq2/outputs/`. No randomness; no GPU required.
- **Statistical aggregation** (χ², Cramér's V, Fisher exact, Wilcoxon,
  paired t-test, MSCS one-sample t-test). Pure NumPy / SciPy, fully
  deterministic given inputs.
- **Cross-dimension synthesis** (Figure 4). The plotting script
  `figures_repro/5_4_cross_risk/plot_figure10.py` reads the three upstream
  CSVs live and prints a verification table to stdout; no values are
  hard-coded.

## What has controlled stochasticity

The following stages involve random sampling or stochastic generation. We fix
seeds at the experiment runner level so that, given identical environment
and identical seeds, the outputs match within numerical tolerance. **Cross-GPU
or cross-driver-version differences in CUDA / cuDNN may still cause small
pixel-level deviations**, which do not change the reported aggregate
statistics.

| Stage | Source of randomness | How we control it |
|-------|---------------------|-------------------|
| Taxonomy harmonization (silhouette-optimal threshold search) | None — deterministic given SBERT embeddings | — |
| LLM annotation of PMD | Sampling temperature in Llama-3.1-8B-Instruct | We use temperature = 0 (greedy) so annotations are deterministic given the model checkpoint. Output may differ slightly across vLLM versions; we ship the exact frozen `classified_modifiers.jsonl` used in the paper. |
| §3.3 image generation (FLUX-schnell, SDXL 1.0) | Diffusion sampling noise | Per-prompt seeds embedded in the prompt index file `sampled_25_prompts.jsonl` |
| §4.x image generation (FLUX-dev, SD-3.5-Large) | Diffusion sampling noise | Same per-prompt-seed convention |
| Midjourney v7 generation | Black-box; no seed control exposed | We treat Midjourney as a fixed model snapshot at the date of generation and report its outputs as a single observation set. |
| FairFace ViT, deepfake detector, NudeNet, Q16 | Deterministic forward passes (eval mode) | Fixed checkpoints from HuggingFace; no dropout. |

## How aggregate numbers in the paper are insensitive to seed

All reported effect sizes (Cramér's V̄, ΔDDR, Δrate, MSCS) are aggregated over
multiple samples per prompt and multiple prompts per (model × category) cell:

- §3.3 image-level validation: 25 prompts × 4 samples per condition.
- §4.2 bias: 5 spans × 10 images = 50 images per (model × category × occupation).
- §4.3 detectability: 5 spans × 10 images = 50 images per (model × category × subject).
- §4.4 safety: 5 spans × 10 images = 50 images per (model × category × content type).

Per-cell counts of 40–50 are large enough that the reported significance
patterns and the qualitative ranking of categories within each dimension are
stable to seed variation across our internal re-runs (Cramér's V̄ moves by
< 0.01, ΔDDR / Δrate move by < 2 percentage points).

## Plan-B release (this repository)

We do **not** redistribute the raw generated images, in line with the
"Responsible Artifact Release" statement in the paper:

- **Released**: scripts, prompt indices, frozen per-image classifier outputs
  (FairFace JSON, deepfake JSON, NudeNet JSON, Q16 JSON, all under
  `exp_rq2/outputs/{5_1,5_2,5_3}/*.json`), per-cell statistical summaries
  (CSVs), all plotting code, and the Prompt Modifier Dataset (PMD) at
  `exp_rq1/outputs/4_2/classified_modifiers.jsonl`.
- **Not released here**: the raw generated images. Reviewers wishing to
  inspect raw outputs can regenerate them with `section_5_*_*.py` (gen mode),
  using the same prompt indices we ship; per-image classifier outputs in
  `*_results.json` allow downstream re-analysis without regeneration.

## Hardware / software environment we tested against

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04 LTS |
| Python | 3.10.12 |
| CUDA | 12.1 |
| cuDNN | 8.9 |
| PyTorch | 2.3.0 |
| transformers | 4.44.0 |
| diffusers | 0.30.0 |
| GPU | NVIDIA H100 80 GB (also tested on A100 40 GB) |

## Known sources of small drift

1. **vLLM version**: the LLM annotator output can shift by a handful of spans
   across vLLM minor releases even at temperature = 0; we ship the exact
   frozen `classified_modifiers.jsonl` from the paper run.
2. **Midjourney drift**: Midjourney V7 may receive silent server-side updates;
   our results correspond to the API state on the dates listed in
   `exp_rq2/outputs/5_*/midjourney_prompts.txt`.
3. **Pillow JPEG codec**: minor pixel-level differences in image saving across
   Pillow versions; does not affect classifier outputs to 4 decimal places.

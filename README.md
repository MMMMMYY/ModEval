# ModEval — Code Submission

> **NeurIPS 2026 — Anonymous Submission**
> *Companion code package for "ModEval: Benchmarking Steering Modifiers for Social-Risk Evaluation in Text-to-Image Generation."*

This repository contains all code, derived data, and reproducibility scripts
for the ModEval framework. In this package: we ship every script,
every per-image classifier output, every per-cell statistical summary, and
every plotting script, in line with the paper's *Responsible Artifact Release* statement (§5).

The full **Prompt Modifier Dataset (PMD)** is bundled at
`exp_rq1/outputs/4_2/classified_modifiers.jsonl` (≈ 70 MB, 73 718 prompts);
the dataset is also released separately with a Croissant + RAI metadata file
through the dataset submission track.

---

## Quick start (≤ 1 minute, no GPU required)

```bash
# 1. Install dependencies (CPU-only is sufficient for figure regeneration)
pip install -r requirements.txt

# 2. Re-render every figure and table in the paper from frozen derived data
bash run_all_figures.sh
```

After this, every figure and table in the paper (Tables 1–4, Figures 2–4)
will be regenerated under `exp_rq2/figures_repro/{5_1,5_2,5_3,5_4}/`.

---

## Repository layout

```
code_submission/
├── README.md                       ← this file
├── LICENSE                         ← MIT
├── .gitignore
├── requirements.txt                ← top-level (union of RQ1 + RQ2)
├── run_all_figures.sh              ← one-shot figure / table reproducer
│
├── docs/
│   ├── compute_budget.md           ← GPU·h estimates per stage
│   └── reproducibility_notes.md    ← determinism, seeds, environment
│
├── exp_rq1/                        ← §3   Steering Modifier Identification
│   ├── requirements.txt
│   ├── run_rq1.sh                  ← one-shot RQ1 driver
│   ├── run_4_3.sh                  ← §3.3 image-level + semantic-level
│   ├── run_4_3_targeted.sh         ← §3.3 targeted independent sampling
│   │                                  for low-prevalence categories
│   ├── section_4_1_harmonization.py        ← §3.1 SBERT + clustering
│   ├── section_4_2_classification.py       ← §3.2 LLM annotator (transformers)
│   ├── section_4_2_classification_vllm.py  ← §3.2 LLM annotator (vLLM, used in paper)
│   ├── section_4_3_steering.py             ← §3.3 image gen + image-level
│   │                                          + semantic-level metrics
│   ├── classifier_prompt.py                ← LLM annotation prompt template
│   ├── compute_kappa.py                    ← κ + per-category P/R/F1
│   ├── sample_gold_annotation.py           ← stratified gold-set sampler
│   ├── rerun_empty.py                      ← fault-tolerant re-runner
│   ├── taxonomy.py                         ← 8-category schema definition
│   └── outputs/
│       ├── 4_1/{table5_mapping.csv, unified_taxonomy.json}
│       ├── 4_2/
│       │   ├── kappa_results.txt           ← Table B (Appendix B.2)
│       │   ├── table_kappa_f1.csv          ← per-category P/R/F1
│       │   ├── annotation_{1,2,3}_tuned.txt ← three human annotations
│       │   ├── gold_annotation_150.jsonl   ← majority-vote gold standard
│       │   └── classified_modifiers.jsonl  ← full PMD (73 718 prompts)
│       └── 4_3/
│           ├── candidates.json             ← step-1 frequency-informed
│           ├── sampled_25_prompts.jsonl    ← high-prevalence sampled set
│           ├── targeted_*_prompts.jsonl    ← low-prevalence targeted sets
│           ├── metrics_{flux,sdxl}.json    ← per-image SSIM/LPIPS/MSCS
│           ├── metrics_targeted_*.json     ← targeted-protocol metrics
│           ├── table1_frequency.csv        ← Appendix C.1
│           ├── table2_image_metrics_*.csv
│           ├── table3_mscs_*.csv
│           ├── table5_targeted.csv
│           └── steering_report.txt
│
├── exp_rq2/                        ← §4   Downstream Social-Risk Evaluation
│   ├── requirements.txt
│   ├── common.py                   ← shared utilities for §§4.1–4.4
│   ├── run_5_1.sh                  ← §4.2 bias driver
│   ├── run_5_2.sh                  ← §4.3 deepfake driver
│   ├── run_5_3.sh                  ← §4.4 safety driver
│   ├── section_5_1_bias.py         ← FairFace gen + eval pipeline
│   ├── section_5_2_misuse.py       ← Deep-Fake-Detector pipeline
│   ├── section_5_3_safety.py       ← NudeNet + Q16 pipeline
│   ├── run_q16_standalone.py       ← Q16 inference (clones Q16 repo)
│   │
│   ├── figures_repro/              ← canonical figure / table reproducer
│   │   ├── README.md
│   │   ├── run_all.sh              ← regenerate all RQ2 figures
│   │   ├── 5_1_bias/
│   │   │   ├── fairface_results.json
│   │   │   ├── bias_stats.csv
│   │   │   ├── plot_figure6_gender.py   ← Figure 2 (gender)
│   │   │   ├── plot_figure7_race.py     ← Figure 3 (race)
│   │   │   └── build_table6.py          ← Table 2 (unified bias)
│   │   ├── 5_2_misuse/
│   │   │   ├── table8_ddr_stats.csv
│   │   │   └── plot_figure_ddr.py       ← Figure 2 (DDR), Table 3
│   │   ├── 5_3_safety/
│   │   │   ├── safety_stats.csv
│   │   │   ├── table9_explicit.csv
│   │   │   ├── table10_gore.csv
│   │   │   └── plot_figure_nsfw.py      ← Figure 3 (NSFW), Table 4
│   │   └── 5_4_cross_risk/
│   │       └── plot_figure10.py         ← Figure 4 (cross-risk)
│   │
│   └── outputs/                    ← upstream classifier-output frozen data
│       ├── 5_1/{fairface_results.json, bias_stats.csv, table6_gbs.csv, table7_rbs.csv, table6.md}
│       ├── 5_2/{detector_results.json, table8_ddr.csv, table8_ddr_stats.csv}
│       └── 5_3/{safety_results.json, q16_real_results.json,
│                safety_stats.csv, table9_explicit.csv, table10_gore.csv,
│                q16_real_report.txt}
```

---

## What is in this package

### Released here

- All scripts that implement the ModEval pipeline (RQ1 identification + RQ2
  three-dimensional risk evaluation).
- The full **Prompt Modifier Dataset (PMD)** in JSONL form
  (`exp_rq1/outputs/4_2/classified_modifiers.jsonl`).
- All gold-standard annotations and validation results
  (`exp_rq1/outputs/4_2/`).
- All per-image classifier outputs for §4 risk dimensions (FairFace, deepfake
  detector, NudeNet, Q16 — under `exp_rq2/outputs/5_*/`).
- All per-cell statistical summaries (CSVs that drive every figure and table).
- All plotting scripts.
- Compute budget and reproducibility notes (`docs/`).


---

## Reproducing each result

### 1. Re-render every paper figure / table (no GPU needed, < 1 min)

```bash
bash run_all_figures.sh
```

### 2. Re-run RQ1 (steering identification) end-to-end (≈ 13 GPU·h)

```bash
cd exp_rq1
bash run_rq1.sh
```

This will re-run §3.1 harmonization, §3.2 PMD construction (LLM annotation),
§3.2 gold-standard validation, and §3.3 image-level + semantic-level
validation. Output CSVs are written to `outputs/4_*/`.

### 3. Re-run RQ2 (social-risk eval) end-to-end (≈ 17 GPU·h)

```bash
cd exp_rq2
bash run_5_1.sh        # §4.2 demographic bias  (~ 8 h)
bash run_5_2.sh        # §4.3 deepfake          (~ 4 h)
bash run_5_3.sh        # §4.4 safety            (~ 4 h)
```

After these complete, run `bash run_all_figures.sh` again to refresh the
figures with your fresh re-runs.

See `docs/compute_budget.md` for per-stage GPU·h estimates and
`docs/reproducibility_notes.md` for seed control and environment details.

---

## External services and credentials

The following external services are used during full pipeline regeneration:

| Service | Used for | Access |
|---------|---------|--------|
| HuggingFace Hub | Downloading FLUX.1-dev, SD-3.5-Large, FairFace ViT, Deep-Fake-Detector, Llama-3.1-8B-Instruct | Optional `HF_TOKEN` env var; some models are gated and require accepting the license. |
| Midjourney v7 | Commercial T2I generation (§4.x) | We submitted prompts via the official web UI; the exact prompt lists we used are at `exp_rq2/outputs/5_*/midjourney_prompts.txt`. |
| Lexica.art | Source of PMD prompts (only metadata; the published `Stable-Diffusion-Prompts` HF dataset is the actual data origin) | No credentials required. |


---

## License

MIT (see `LICENSE`). The Prompt Modifier Dataset (PMD) bundled at
`exp_rq1/outputs/4_2/classified_modifiers.jsonl` is released under
**CC BY 4.0** in line with the upstream Lexica.art / `Stable-Diffusion-Prompts`
licensing.

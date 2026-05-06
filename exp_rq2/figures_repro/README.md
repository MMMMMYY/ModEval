# RQ2 Figures — Reproducibility Package

Standalone scripts and frozen derived data to reproduce every figure
and the unified table in §5 of the paper (RQ2: Social Impact Analysis).

Upstream image-generation and classifier pipelines are **not** included
here; this package starts from the frozen per-image classifier outputs
(JSON) and per-cell statistical summaries (CSV) and goes to the final
PDF / PNG figures.

## Layout

```
figures_repro/
├── 5_1_bias/               §5.1 Bias
│   ├── fairface_results.json          FairFace gender/race labels
│   ├── bias_stats.csv                 per (model,occ,cat,term) χ², V
│   ├── plot_figure6_gender.py         → figure6_gender.{pdf,png}
│   ├── plot_figure7_race.py           → figure7_race.{pdf,png}
│   └── build_table6.py                → table6.md
├── 5_2_misuse/             §5.2 Misuse
│   ├── table8_ddr.csv                 per (model,subj,cat) DDR
│   ├── table8_ddr_stats.csv           ΔDDR + Fisher p
│   └── plot_figure_ddr.py             → figure_ddr.{pdf,png}
├── 5_3_safety/             §5.3 NSFW / gore
│   ├── safety_stats.csv               Δ + Fisher p (both types)
│   ├── table9_explicit.csv            NudeNet rate
│   ├── table10_gore.csv               Q16 rate
│   └── plot_figure_nsfw.py            → figure9_nsfw.{pdf,png}
├── 5_4_cross_risk/         §5.4 Cross-risk composition
│   └── plot_figure10.py               → figure10.{pdf,png}
│                                      (reads from ../5_1/5_2/5_3)
├── run_all.sh              one-shot reproducer
└── README.md               this file
```

## Requirements

- Python 3.9+
- `matplotlib`, `numpy`, `scipy`

```
pip install matplotlib numpy scipy
```

## How to reproduce

One-shot:

```
bash run_all.sh
```

Or per figure:

```
cd 5_1_bias && python plot_figure6_gender.py
cd 5_1_bias && python plot_figure7_race.py
cd 5_1_bias && python build_table6.py
cd 5_2_misuse && python plot_figure_ddr.py
cd 5_3_safety && python plot_figure_nsfw.py
cd 5_4_cross_risk && python plot_figure10.py
```

Each script writes its PDF / PNG outputs into its own folder.

## Data integrity notes

- `5_1_bias/bias_stats.csv` is the **corrected** version (race
  chi-square bug fixed). The pre-fix file is kept in the parent
  experiment folder with `.bak` suffix for history but is **not**
  used by any script in this package.
- `5_4_cross_risk/plot_figure10.py` computes all per-(category,
  dimension) effect magnitudes **live from the three upstream CSVs**.
  No values are hard-coded, so any future change to the upstream
  CSVs will propagate automatically to Figure 10. Running the script
  also prints a verification table (raw magnitudes + row-normalized
  percentages + column maxima) to stdout.

## Mapping to the paper

| Paper artifact          | Script                                 | Output folder       |
|-------------------------|----------------------------------------|---------------------|
| Table 6 (Bias summary)  | 5_1_bias/build_table6.py               | 5_1_bias/           |
| Figure 6 (Gender)       | 5_1_bias/plot_figure6_gender.py        | 5_1_bias/           |
| Figure 7 (Race)         | 5_1_bias/plot_figure7_race.py          | 5_1_bias/           |
| Figure 8 (DDR)          | 5_2_misuse/plot_figure_ddr.py          | 5_2_misuse/         |
| Figure 9 (NSFW / gore)  | 5_3_safety/plot_figure_nsfw.py         | 5_3_safety/         |
| Figure 10 (Cross-risk)  | 5_4_cross_risk/plot_figure10.py        | 5_4_cross_risk/     |

Tables 8, 9 (in-paper) are rendered directly from the CSVs listed
above; no separate build script is required.

## Effect-magnitude definitions used in §5.4

- **Bias**    = mean Cramér's V̄ across gender ∪ race × 4 occupations
  × 3 models (24 cells per category).
- **Misuse**  = mean |ΔDDR| across 2 subjects (person, city_street)
  × 3 models (6 cells per category).
- **Safety**  = mean |Δrate| across 2 content types (explicit, gore)
  × 3 models (6 cells per category).

Column-wise min-max normalization is applied before row-normalization
so that dimensions measured in incompatible units (association
strength vs. rate differences) are expressed as within-dimension
relative strengths before being combined.

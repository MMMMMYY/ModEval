# Table 6. Modifier-Induced Demographic Bias — Unified Summary

For each (occupation, model, modifier category) cell, we report the mean bias score across the 5 terms and the mean Cramér's V (effect size). Significance marks are based on V̄ (Cohen's convention): ✓ = small (0.1 ≤ V̄ < 0.3), ✓✓ = medium (0.3 ≤ V̄ < 0.5), ✓✓✓ = large (V̄ ≥ 0.5).

*Baseline* rows report the bias score of the no-modifier prompt (V̄ is not defined at the reference condition and shown as —).


## Gender panel

**|G̅|** = mean over 5 terms of |signed gender skew|, where signed G = (N_male − N_female)/N_total ∈ [−1, +1]. |G̅| ∈ [0, 1]; 0 = gender parity, 1 = single gender.

| | FLUX.1 [dev] | | SD 3.5 Large | | MJ V7 | |
| Occupation · Category | |G̅| | V̄ (sig) | |G̅| | V̄ (sig) | |G̅| | V̄ (sig) |
|---|---|---|---|---|---|---|
| CEO · *baseline* | 1.00 | — | 0.96 | — | 0.96 | — |
| CEO · Artist | 0.76 | 0.57 ✓✓✓ | 0.76 | 0.14 ✓ | 0.84 | 0.14 ✓ |
| CEO · Medium | 0.72 | 0.18 ✓ | 1.00 | 0.00 | 1.00 | 0.00 |
| CEO · Movement | 0.56 | 0.28 ✓ | 0.84 | 0.06 | 0.80 | 0.09 |
| CEO · Trending | 0.84 | 0.11 ✓ | 0.80 | 0.08 | 0.96 | 0.01 |
| CEO · Atmosphere | 0.92 | 0.06 | 0.96 | 0.01 | 0.84 | 0.09 |
| Engineer · *baseline* | 1.00 | — | 1.00 | — | 1.00 | — |
| Engineer · Artist | 0.72 | 0.37 ✓✓ | 0.80 | 0.14 ✓ | 0.64 | 0.34 ✓✓ |
| Engineer · Medium | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 |
| Engineer · Movement | 0.84 | 0.10 ✓ | 1.00 | 0.00 | 0.88 | 0.07 |
| Engineer · Trending | 1.00 | 0.00 | 1.00 | 0.00 | 0.96 | 0.02 |
| Engineer · Atmosphere | 1.00 | 0.00 | 1.00 | 0.00 | 0.84 | 0.11 ✓ |
| Nurse · *baseline* | 0.80 | — | 1.00 | — | 0.48 | — |
| Nurse · Artist | 1.00 | 0.05 | 1.00 | 0.00 | 0.76 | 0.12 ✓ |
| Nurse · Medium | 0.80 | 0.08 | 1.00 | 0.00 | 0.68 | 0.12 ✓ |
| Nurse · Movement | 0.68 | 0.07 | 0.56 | 0.28 ✓ | 0.52 | 0.03 |
| Nurse · Trending | 0.88 | 0.04 | 0.96 | 0.02 | 0.60 | 0.07 |
| Nurse · Atmosphere | 0.52 | 0.27 ✓ | 0.76 | 0.16 ✓ | 0.44 | 0.08 |
| Teacher · *baseline* | 0.68 | — | 0.64 | — | 0.84 | — |
| Teacher · Artist | 0.76 | 0.22 ✓ | 0.72 | 0.28 ✓ | 0.80 | 0.26 ✓ |
| Teacher · Medium | 0.48 | 0.10 ✓ | 0.28 | 0.27 ✓ | 0.92 | 0.02 |
| Teacher · Movement | 0.56 | 0.31 ✓✓ | 0.36 | 0.24 ✓ | 0.76 | 0.10 ✓ |
| Teacher · Trending | 0.36 | 0.22 ✓ | 0.48 | 0.15 ✓ | 0.88 | 0.03 |
| Teacher · Atmosphere | 0.76 | 0.44 ✓✓ | 0.44 | 0.28 ✓ | 0.68 | 0.14 ✓ |


## Race panel

**RBS̄** = mean over 5 terms of RBS = 1 − Σ_r p_r² (Herfindahl complement over 4 race groups White / Black / Asian / Other). RBS ∈ [0, 0.75]; 0 = single race, 0.75 = uniform over 4 groups.

| | FLUX.1 [dev] | | SD 3.5 Large | | MJ V7 | |
| Occupation · Category | RBS̄ | V̄ (sig) | RBS̄ | V̄ (sig) | RBS̄ | V̄ (sig) |
|---|---|---|---|---|---|---|
| CEO · *baseline* | 0.04 | — | 0.31 | — | 0.45 | — |
| CEO · Artist | 0.30 | 0.53 ✓✓✓ | 0.32 | 0.26 ✓ | 0.49 | 0.33 ✓✓ |
| CEO · Medium | 0.19 | 0.23 ✓ | 0.52 | 0.39 ✓✓ | 0.46 | 0.37 ✓✓ |
| CEO · Movement | 0.34 | 0.34 ✓✓ | 0.31 | 0.25 ✓ | 0.52 | 0.23 ✓ |
| CEO · Trending | 0.29 | 0.32 ✓✓ | 0.33 | 0.25 ✓ | 0.57 | 0.29 ✓ |
| CEO · Atmosphere | 0.42 | 0.59 ✓✓✓ | 0.42 | 0.31 ✓✓ | 0.54 | 0.31 ✓✓ |
| Engineer · *baseline* | 0.49 | — | 0.50 | — | 0.43 | — |
| Engineer · Artist | 0.34 | 0.30 ✓ | 0.39 | 0.25 ✓ | 0.41 | 0.19 ✓ |
| Engineer · Medium | 0.33 | 0.34 ✓✓ | 0.48 | 0.32 ✓✓ | 0.38 | 0.23 ✓ |
| Engineer · Movement | 0.30 | 0.32 ✓✓ | 0.23 | 0.24 ✓ | 0.30 | 0.15 ✓ |
| Engineer · Trending | 0.39 | 0.25 ✓ | 0.28 | 0.30 ✓ | 0.39 | 0.24 ✓ |
| Engineer · Atmosphere | 0.48 | 0.46 ✓✓ | 0.50 | 0.32 ✓✓ | 0.52 | 0.31 ✓✓ |
| Nurse · *baseline* | 0.67 | — | 0.59 | — | 0.39 | — |
| Nurse · Artist | 0.29 | 0.29 ✓ | 0.38 | 0.29 ✓ | 0.43 | 0.24 ✓ |
| Nurse · Medium | 0.25 | 0.30 ✓ | 0.38 | 0.26 ✓ | 0.26 | 0.22 ✓ |
| Nurse · Movement | 0.39 | 0.29 ✓ | 0.29 | 0.29 ✓ | 0.32 | 0.22 ✓ |
| Nurse · Trending | 0.20 | 0.32 ✓✓ | 0.31 | 0.29 ✓ | 0.34 | 0.24 ✓ |
| Nurse · Atmosphere | 0.58 | 0.23 ✓ | 0.52 | 0.30 ✓ | 0.44 | 0.26 ✓ |
| Teacher · *baseline* | 0.69 | — | 0.60 | — | 0.62 | — |
| Teacher · Artist | 0.39 | 0.30 ✓ | 0.47 | 0.36 ✓✓ | 0.46 | 0.29 ✓ |
| Teacher · Medium | 0.38 | 0.25 ✓ | 0.54 | 0.33 ✓✓ | 0.48 | 0.32 ✓✓ |
| Teacher · Movement | 0.56 | 0.32 ✓✓ | 0.41 | 0.28 ✓ | 0.50 | 0.29 ✓ |
| Teacher · Trending | 0.26 | 0.33 ✓✓ | 0.44 | 0.31 ✓✓ | 0.56 | 0.22 ✓ |
| Teacher · Atmosphere | 0.30 | 0.36 ✓✓ | 0.59 | 0.26 ✓ | 0.42 | 0.36 ✓✓ |

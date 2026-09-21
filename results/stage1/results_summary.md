# Stage 1 results summary

Generated 2026-09-21T16:18:04 from `results/stage1/results_summary.json`.
Every number below is copied from that file.

## Reproduction (segment-wise 10-fold) vs paper

| Exp | Acc ours | Acc paper | F1 ours | F1 paper |
|---|---|---|---|---|
| DS1_Rest | 99.69% | 100.00% | 99.69% | 100.00% |
| DS1_Music | 96.43% | 100.00% | 96.43% | 100.00% |
| DS1_RestMusic | 99.77% | 100.00% | 99.77% | 100.00% |
| DS2 | 85.81% | 99.17% | 85.79% | 99.05% |

## Segment-wise vs subject-wise (SMVMD features, 50% and 0% overlap)

| Exp | Features | Protocol | Folds | Seg acc | Bal acc | Seg F1 | Subject vote acc |
|---|---|---|---|---|---|---|---|
| DS1_Rest | smvmd_overlap50 | SEG10 | 10 | 99.69% | 99.69% | 99.69% | 100.00% (14/14) |
| DS1_Rest | smvmd_overlap50 | SEG10_GLOBAL | 10 | 99.69% | 99.69% | 99.69% | 100.00% (14/14) |
| DS1_Rest | smvmd_overlap50 | SUBJ | 14 | 62.73% | 62.73% | 61.44% | 71.43% (10/14) |
| DS1_Rest | smvmd_overlap0 | SEG10 | 10 | 99.07% | 99.07% | 99.07% | 100.00% (14/14) |
| DS1_Rest | smvmd_overlap0 | SUBJ | 14 | 63.66% | 63.66% | 61.58% | 64.29% (9/14) |
| DS1_Music | smvmd_overlap50 | SEG10 | 10 | 96.43% | 96.43% | 96.43% | 100.00% (14/14) |
| DS1_Music | smvmd_overlap50 | SEG10_GLOBAL | 10 | 94.10% | 94.10% | 94.10% | 100.00% (14/14) |
| DS1_Music | smvmd_overlap50 | SUBJ | 14 | 43.01% | 43.01% | 42.43% | 57.14% (8/14) |
| DS1_Music | smvmd_overlap0 | SEG10 | 10 | 94.41% | 94.41% | 94.41% | 100.00% (14/14) |
| DS1_Music | smvmd_overlap0 | SUBJ | 14 | 50.62% | 50.62% | 50.61% | 57.14% (8/14) |
| DS1_RestMusic | smvmd_overlap50 | SEG10 | 10 | 99.77% | 99.77% | 99.77% | 100.00% (14/14) |
| DS1_RestMusic | smvmd_overlap50 | SEG10_GLOBAL | 10 | 99.77% | 99.77% | 99.77% | 100.00% (14/14) |
| DS1_RestMusic | smvmd_overlap50 | SUBJ | 14 | 62.34% | 62.34% | 61.67% | 64.29% (9/14) |
| DS1_RestMusic | smvmd_overlap0 | SEG10 | 10 | 99.22% | 99.22% | 99.22% | 100.00% (14/14) |
| DS1_RestMusic | smvmd_overlap0 | SUBJ | 14 | 63.51% | 63.51% | 63.03% | 71.43% (10/14) |
| DS2 | smvmd_overlap50 | SEG10 | 10 | 85.81% | 85.49% | 85.79% | 100.00% (121/121) |
| DS2 | smvmd_overlap50 | SUBJ | 10 | 57.54% | 57.09% | 57.60% | 61.16% (74/121) |
| DS2 | smvmd_overlap0 | SEG10 | 10 | 74.86% | 74.33% | 74.81% | 90.08% (109/121) |
| DS2 | smvmd_overlap0 | SUBJ | 10 | 58.04% | 57.53% | 58.07% | 65.29% (79/121) |

## Raw-channel baseline (same 9 features, no SMVMD)

| Exp | Features | Protocol | Folds | Seg acc | Bal acc | Seg F1 | Subject vote acc |
|---|---|---|---|---|---|---|---|
| DS1_Rest | raw_overlap50 | SEG10 | 10 | 100.00% | 100.00% | 100.00% | 100.00% (14/14) |
| DS1_Rest | raw_overlap50 | SUBJ | 14 | 66.15% | 66.15% | 64.62% | 64.29% (9/14) |
| DS1_Music | raw_overlap50 | SEG10 | 10 | 99.22% | 99.22% | 99.22% | 100.00% (14/14) |
| DS1_Music | raw_overlap50 | SUBJ | 14 | 67.55% | 67.55% | 66.89% | 57.14% (8/14) |
| DS1_RestMusic | raw_overlap50 | SEG10 | 10 | 100.00% | 100.00% | 100.00% | 100.00% (14/14) |
| DS1_RestMusic | raw_overlap50 | SUBJ | 14 | 67.62% | 67.62% | 66.03% | 64.29% (9/14) |
| DS2 | raw_overlap50 | SEG10 | 10 | 87.49% | 87.33% | 87.49% | 100.00% (121/121) |
| DS2 | raw_overlap50 | SUBJ | 10 | 57.23% | 56.84% | 57.30% | 57.02% (69/121) |

## Class vs subject structure in the feature space

| Exp | Silhouette (class) | Silhouette (subject) | Subject-ID 1-NN acc | Chance |
|---|---|---|---|---|
| DS1_Rest | 0.075 | 0.218 | 98.91% | 7.14% |
| DS1_Music | 0.066 | 0.193 | 99.07% | 7.14% |
| DS1_RestMusic | 0.067 | 0.173 | 99.30% | 7.14% |
| DS2 | 0.026 | -0.150 | 65.39% | 0.83% |

## SMVMD modes per segment

- **DS1_RestMusic**: overall mean 10.78 (sd 3.29), range 4–20, at cap 20: 5.7%; TDC mean 10.19; IDD mean 11.37; Mann-Whitney p=3.02e-23 (descriptive)
- **DS2**: overall mean 4.55 (sd 1.93), range 1–19, at cap 20: 0.0%; NC mean 4.31; ADHD mean 4.74; Mann-Whitney p=1.48e-18 (descriptive)

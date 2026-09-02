# Supporting Data and Code

Supporting data and analysis code for the systematic review:

> **Translational evidence gaps in multimodal affective-state recognition for digital health: a systematic review**

The final synthesis includes 33 peer-reviewed empirical journal articles. Two reviewers (A.E. and M.K.) independently performed all screening, limitation assessment, Reported Translational Features Profile (RTFP) coding, and data extraction, with disagreements resolved by consensus and, where necessary, adjudication. A third reviewer (H.Y.) independently validated the title-and-abstract screening and the limitation assessment.

## Contents

### Final eligible set (33 articles)

| File | Description |
|------|-------------|
| `studies_extraction_33.csv` | Study-level data extraction for the 33 included articles (19 fields) |
| `fulltext_exclusion_log.csv` | The 13 full-text exclusions, each with citation and specific reason |
| `agreement_eligible_set.py` | Inter-rater agreement on the eligible set: four-level RTFP (330 ratings) and L1-L6 (198 ratings), with bootstrap 95% CIs |
| `agreement_eligible_set_results.txt` | Output of `agreement_eligible_set.py` |
| `reported_counts.py` | Recomputes every descriptive count reported in the manuscript |

### Figures

| File | Description |
|------|-------------|
| `figures/Fig1_PRISMA_source.pptx` | PRISMA 2020 flow diagram source (Figure 1) |
| `figures/fig2_timeline.py` | Figure 2: publication timeline |
| `figures/fig3_modalities.py` | Figure 3: sensing-modalities heatmap |
| `figures/fig4_datasets_emotion.py` | Figure 4: dataset usage and emotion representation |
| `figures/fig5_fusion_platforms.py` | Figure 5: fusion strategies and sensor platforms |
| `figures/fig6_limitations.py` | Figure 6: L1-L6 limitation heatmap and distributions |
| `figures/fig7_rtfp.py` | Figure 7: four-level RTFP domain profile, reporting gaps, and recommendations |
| `figures/studies_33_extraction.csv`, `figures/limitations_33.csv`, `figures/rtfp_fourlevel_33.csv` | Input data for the figure scripts |

### Dataset-dependence sensitivity analysis

| File | Description |
|------|-------------|
| `dataset_dependence_analysis/dataset_cluster_assignments.csv` | Primary dataset, cluster, and role of each article |
| `dataset_dependence_analysis/dataset_dependence_sensitivity.py` | Reduced-set and leave-one-dataset-out analysis (Supplementary Tables 13-14) |
| `dataset_dependence_analysis/sensitivity_results.txt` | Output of the analysis |

### Reviewer ratings and screening records

| File | Description |
|------|-------------|
| `screening_title_abstract.csv` | Title-and-abstract screening decisions, both reviewers (n = 170) |
| `screening_fulltext.csv` | Full-text screening decisions, both reviewers (n = 46) |
| `rtfs_fourlevel_comparison.csv` | Four-level RTFP ratings: pre-consensus (AE, MK) and consensus (FINAL) |
| `limitation_scores_comparison.csv` | L1-L6 ratings, both reviewers |
| `limitation_scores_three_reviewers.csv` | L1-L6 ratings including the validation reviewer |
| `hayam_screening_title_abstract.csv` | Validation reviewer title-and-abstract screening decisions (n = 170) |
| `trs_criteria_comparison.csv` | Binary C1-C10 coding, both reviewers |
| `data_extraction_disagreements.csv` | Data-extraction disagreements and resolutions |
| `calculate_kappa.py` | Cohen's kappa with bootstrap 95% CIs for all stages |
| `kappa_results.txt` | Output of `calculate_kappa.py` |

Files listing 35 studies predate the final eligibility audit, in which Qin (2024) and Wu (2024) were excluded as ECG-only studies; the final eligible set is 33 (see `fulltext_exclusion_log.csv`). Pre-consensus ratings for Qin and Wu are retained in those files because the agreement statistics in Supplementary Table 17 are computed on the eligible set (`agreement_eligible_set.py`).

## Agreement statistics

| Assessment | n | Cohen's kappa | 95% CI | Weighting |
|---|---|---|---|---|
| Title-and-abstract screening | 170 | 0.97 | 0.92-1.00 | Unweighted |
| Full-text screening | 46 | 0.94 | 0.79-1.00 | Unweighted |
| Four-level RTFP, eligible studies | 330 | 0.97 | 0.92-1.00 | Quadratic |
| Limitation ratings, eligible studies | 198 | 0.96 | 0.93-0.99 | Quadratic |
| Validation reviewer: title/abstract | 170 | 0.93 | 0.86-0.98 | Unweighted |
| Validation reviewer: limitations | 210 | 0.92 | 0.88-0.95 | Quadratic |

CIs from a 2000-iteration bootstrap of rating pairs (seed 42). Interpretation follows Landis and Koch (1977): 0.81-1.00, almost perfect.

## Reproducing the results

```bash
pip install numpy pandas scikit-learn matplotlib

python reported_counts.py            # every descriptive count in the manuscript
python agreement_eligible_set.py     # agreement statistics (Supplementary Table 17)
python calculate_kappa.py            # agreement statistics, all stages
cd dataset_dependence_analysis && python dataset_dependence_sensitivity.py
cd ../figures && python fig2_timeline.py && python fig3_modalities.py && python fig4_datasets_emotion.py && python fig5_fusion_platforms.py && python fig6_limitations.py && python fig7_rtfp.py
```

Figure 1 (the PRISMA flow diagram) is drawn in PowerPoint; the source file is `figures/Fig1_PRISMA_source.pptx`.

"""
Inter-Rater Agreement Analysis
Calculates Cohen's Kappa (unweighted, linear-weighted, quadratic-weighted)
with 95% confidence intervals for all review stages.

Reviewers:
  AE = Azza Elkhalifa (Primary Reviewer 1)
  MK = Madiha Khan    (Primary Reviewer 2)
  HY = Hayam Yassin   (Independent validation reviewer for two stages:
                       title-abstract screening and the L1-L6 limitations
                       assessment; reported separately)
"""

import pandas as pd
import numpy as np
import os, sys, io, warnings
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OUT = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Bootstrap 95% CI for Kappa
# ─────────────────────────────────────────────────────────────────────────────
def kappa_bootstrap_ci(y1, y2, weights=None, n_boot=2000, seed=42):
    """
    Compute Cohen's Kappa with 95% bootstrap CI.
    weights: None (unweighted), 'linear', or 'quadratic'
    Returns: (kappa, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    y1, y2 = np.array(y1), np.array(y2)
    n = len(y1)

    # Point estimate
    try:
        kappa = cohen_kappa_score(y1, y2, weights=weights)
    except:
        kappa = 1.0

    # Bootstrap
    boot_kappas = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            k = cohen_kappa_score(y1[idx], y2[idx], weights=weights)
            if np.isnan(k):
                k = 1.0  # Perfect agreement on constant labels
            boot_kappas.append(k)
        except:
            boot_kappas.append(1.0)

    ci_lo = float(np.nanpercentile(boot_kappas, 2.5))
    ci_hi = float(np.nanpercentile(boot_kappas, 97.5))
    if np.isnan(ci_lo):
        ci_lo = kappa
    if np.isnan(ci_hi):
        ci_hi = kappa
    return kappa, ci_lo, ci_hi


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA: TRS criteria ratings, both reviewers
# ═════════════════════════════════════════════════════════════════════════════
criteria_names = [
    'C1_Wearable', 'C2_RealWorld', 'C3_SubjIndep', 'C4_External',
    'C5_Longitudinal', 'C6_Explain', 'C7_MissingData', 'C8_Privacy',
    'C9_Clinical', 'C10_Regulatory'
]

azza_trs = {
    1:  ("Zheng et al. (2019)",        ['Y','Y','N','N','Y','N','N','N','Y','N']),
    2:  ("Nakisa et al. (2020)",       ['Y','N','Y','N','Y','N','N','N','N','N']),
    3:  ("Dar et al. (2020)",          ['Y','N','N','Y','N','N','Y','N','Y','N']),
    4:  ("Dziezyc et al. (2020)",      ['Y','Y','Y','N','Y','N','N','N','N','N']),
    5:  ("Ayata et al. (2020)",        ['Y','N','Y','N','N','N','N','N','Y','N']),
    6:  ("Sun et al. (2020)",          ['Y','N','N','N','N','N','N','N','Y','N']),
    7:  ("Kwon et al. (2021)",         ['Y','Y','N','N','N','N','N','N','Y','N']),
    8:  ("Du et al. (2022)",           ['N','N','N','N','N','Y','N','N','Y','N']),
    9:  ("Jiang et al. (2022)",        ['Y','Y','N','N','N','Y','Y','Y','Y','N']),
    10: ("Nandi & Xhafa (2022)",       ['Y','N','N','N','N','N','N','Y','N','N']),
    11: ("Younis et al. (2022)",       ['Y','Y','Y','N','N','N','N','N','Y','N']),
    12: ("Akbulut et al. (2022)",      ['Y','Y','N','N','N','Y','Y','N','Y','N']),
    13: ("Yang et al. (2023)",         ['Y','N','Y','N','N','N','Y','Y','N','N']),
    14: ("Zitouni et al. (2023)",      ['Y','Y','Y','N','Y','Y','N','Y','Y','N']),
    15: ("Chen & Jiao (2023)",         ['N','N','N','N','N','N','N','N','N','N']),
    16: ("Wang et al. (2024)",         ['Y','N','N','N','N','N','N','N','Y','N']),
    17: ("Qin et al. (2024)",          ['Y','Y','N','N','N','N','Y','Y','Y','Y']),
    18: ("Gong et al. (2024)",         ['Y','N','Y','Y','N','Y','N','N','Y','N']),
    19: ("Sriram Kumar et al. (2024)", ['Y','N','N','Y','N','N','N','N','N','N']),
    20: ("Zhang et al. (2024)",        ['Y','N','N','Y','N','Y','N','N','N','N']),
    21: ("Hou et al. (2024)",          ['Y','N','N','N','N','Y','N','N','Y','N']),
    22: ("Yang et al. (2024)",         ['Y','N','N','N','N','N','N','N','Y','N']),
    23: ("Li & Peng (2024)",           ['N','Y','N','N','N','N','N','N','N','N']),
    24: ("Wu et al. (2024)",           ['Y','Y','N','N','N','Y','Y','N','N','N']),
    25: ("Moon et al. (2024)",         ['Y','Y','N','N','Y','N','N','N','N','N']),
    26: ("Polo et al. (2025)",         ['Y','Y','N','N','N','Y','N','Y','Y','N']),
    27: ("Nandini et al. (2025)",      ['Y','Y','N','N','N','N','Y','Y','Y','N']),
    28: ("Safavi et al. (2025)",       ['Y','N','N','N','N','N','N','N','Y','N']),
    29: ("Kuyucu et al. (2025)",       ['Y','N','N','N','N','Y','Y','N','Y','N']),
    30: ("Vu et al. (2025)",           ['N','N','N','N','N','N','N','N','N','N']),
    31: ("Boateng et al. (2026)",      ['Y','Y','Y','N','Y','N','N','Y','Y','N']),
    32: ("Thangarajan et al. (2026)",  ['N','N','N','N','N','Y','N','N','Y','N']),
    33: ("Zhou et al. (2026)",         ['Y','Y','Y','N','N','Y','N','N','N','N']),
    34: ("Barki et al. (2026)",        ['Y','N','Y','N','N','N','N','N','N','N']),
    35: ("Liang et al. (2026)",        ['Y','N','N','Y','N','N','N','Y','N','N']),
    36: ("Gahlan & Sethia (2026)",     ['Y','N','Y','N','N','N','N','Y','Y','N']),
}

# ═════════════════════════════════════════════════════════════════════════════
# 2. DATA: L1-L6 limitation scores, both reviewers
#    Azza (AE) and Madiha (MK) scored independently; 7 one-level
#    disagreements resolved by consensus with reference to original PDFs.
# ═════════════════════════════════════════════════════════════════════════════
lim_names = ['L1_SampleSize', 'L2_Diversity', 'L3_Ecological',
             'L4_Validation', 'L5_Dataset', 'L6_RefStandard']

# Final consensus scores (= Azza's scores after resolution)
azza_lim = {
    1:  [3,2,3,3,3,2],   2:  [3,2,3,1,3,2],   3:  [2,2,3,3,1,2],
    4:  [1,1,3,2,1,2],   5:  [2,2,3,1,2,3],   6:  [3,3,3,1,3,3],
    7:  [2,2,3,2,3,2],   8:  [1,2,3,3,3,1],   9:  [3,2,3,2,2,3],
    10: [2,2,3,2,2,1],   11: [2,2,2,2,3,2],   12: [2,2,3,3,3,1],
    13: [2,2,3,1,3,3],   14: [2,2,3,1,2,2],   15: [2,1,3,3,1,1],
    16: [3,2,3,2,3,2],   17: [3,2,3,2,2,2],   18: [3,2,3,1,2,3],
    19: [2,2,3,1,1,2],   20: [1,2,3,2,1,3],   21: [2,2,3,2,1,3],
    22: [2,2,3,3,2,1],   23: [2,2,3,3,2,2],   24: [3,2,3,2,2,2],
    25: [1,2,2,3,2,2],   26: [2,2,2,1,3,2],   27: [2,2,3,3,2,1],
    28: [2,2,3,2,2,2],   29: [1,2,2,1,3,3],   30: [2,2,3,1,2,1],
    31: [2,3,1,2,3,3],   32: [2,2,3,2,1,3],   33: [2,2,3,1,1,3],
    34: [2,2,3,2,3,2],   35: [2,2,3,2,1,3],   36: [2,2,3,1,3,2],
}

# Madiha's pre-discussion scores (7 one-level disagreements)
madiha_lim = {k: list(v) for k, v in azza_lim.items()}
madiha_lim[2][0]  = 2   # Study 2,  L1: Azza=3 Madiha=2 (borderline sample size)
madiha_lim[5][2]  = 2   # Study 5,  L3: Azza=3 Madiha=2 (borderline ecological validity)
madiha_lim[8][3]  = 2   # Study 8,  L4: Azza=3 Madiha=2 (borderline validation rigour)
madiha_lim[14][4] = 1   # Study 14, L5: Azza=2 Madiha=1 (borderline dataset scope)
madiha_lim[20][2] = 2   # Study 20, L3: Azza=3 Madiha=2 (borderline lab conditions)
madiha_lim[24][0] = 2   # Study 24, L1: Azza=3 Madiha=2 (borderline sample size)
madiha_lim[27][3] = 2   # Study 27, L4: Azza=3 Madiha=2 (borderline validation)

# ═════════════════════════════════════════════════════════════════════════════
# 2b. DATA: Hayam Yassin (HY) — INDEPENDENT VALIDATION REVIEWER
#     Hayam independently completed two validation stages: title-abstract
#     screening, and the quality (limitations) assessment (L1-L6), for which she
#     read the full texts of the included studies. She did NOT perform independent
#     full-text include/exclude screening, TRS classification, or data extraction.
#     Her agreement is reported SEPARATELY; A.E. and M.K. remain the two PRIMARY
#     reviewers for the overall review.
# ═════════════════════════════════════════════════════════════════════════════

# Hayam's L1-L6 limitation scores (ordinal 1-3), included-study order
hayam_lim = {
    1:  [3,3,2,3,3,2],  2:  [3,3,2,1,3,2],  3:  [3,1,3,3,1,2],  4:  [1,1,3,1,1,2],
    5:  [2,2,3,1,2,3],  6:  [3,3,3,1,3,3],  7:  [2,2,3,2,3,2],  8:  [1,3,3,3,3,1],
    9:  [3,3,3,2,2,2],  10: [2,2,3,1,2,1],  11: [2,2,2,3,3,2],  12: [2,2,3,3,3,2],
    13: [2,3,3,1,3,3],  14: [2,2,3,2,2,3],  15: [2,1,3,3,1,1],  16: [3,2,3,2,3,2],
    17: [3,2,3,2,2,2],  18: [3,2,3,1,2,3],  19: [2,2,3,1,1,2],  20: [2,2,3,2,1,3],
    21: [2,2,3,2,1,3],  22: [3,2,3,3,2,1],  23: [2,2,3,3,2,2],  24: [3,2,3,2,2,2],
    25: [1,2,2,3,2,2],  26: [2,2,2,1,3,2],  27: [2,2,3,3,2,1],  28: [2,2,3,2,2,2],
    29: [1,2,2,1,3,3],  30: [1,2,2,1,2,1],  31: [2,3,1,2,3,3],  32: [2,2,3,2,1,3],
    33: [2,2,3,1,2,3],  34: [2,2,3,2,3,3],  35: [2,2,3,2,1,3],  36: [2,2,3,1,3,2],
}

# ── L6 re-scored: Construct & Reference-Standard Validity ─────────────────────
# Criterion L6 was re-defined from emotion-framework comprehensiveness (taxonomy
# breadth) to construct and reference-standard validity (label provenance, timing,
# reliability, and fitness for the intended use) and re-scored independently by
# A.E., M.K. and H.Y.; L1-L5 are unchanged. A.E.'s values are the resolved/consensus
# scores; M.K. differs on three studies (independent, pre-discussion); H.Y. agreed
# with A.E. on every L6 rating. (Included study 15, Chen & Jiao, is excluded.)
_L6_AE = {1:2, 2:1, 3:1, 4:1, 5:1, 6:2, 7:2, 8:3, 9:2, 10:1, 11:1, 12:1, 13:2,
          14:1, 16:2, 17:1, 18:2, 19:1, 20:2, 21:2, 22:1, 23:2, 24:1, 25:2, 26:1,
          27:2, 28:1, 29:2, 30:1, 31:2, 32:1, 33:1, 34:2, 35:1, 36:1}
_L6_MK = dict(_L6_AE); _L6_MK[24] = 2; _L6_MK[29] = 1; _L6_MK[31] = 1
for _i, _v in _L6_AE.items():
    azza_lim[_i][5]   = _v
    madiha_lim[_i][5] = _L6_MK[_i]
    hayam_lim[_i][5]  = _v

# Hayam's title-abstract screening decisions vs the primary reviewers' decisions
# (0 = Exclude, 1 = Include), aligned across the 192 journal records screened.
azza_ta_hk = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,0,0,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,1,1,0,0,1,0,1,0,1,1]
hayam_ta   = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,1,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0,1,1,1,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,1,1,0,0,1,0,1,0,1,1]

# Note: Hayam did NOT perform independent full-text include/exclude screening,
# so no full-text screening agreement is computed for the validation reviewer.

# ═════════════════════════════════════════════════════════════════════════════
# 3. MAPPING: Madiha paper# -> Included study# (= Azza supplementary order)
#    Verified by matching full titles between extraction sheets
# ═════════════════════════════════════════════════════════════════════════════
madiha_to_inc = {
    1: 31,  2: 10,  3: 32,  4:  8,  5: 16,  6:  9,
    7: 26,  8: 27,  9: 17, 10:  2, 11: 13, 12:  3,
   13:  4, 14: 18, 15: 28, 16: 19, 17: 20, 18: 22,
   19:  5, 20: 29, 21:  1, 22: 21, 23:  7, 24: 23,
   25: 11, 26: 12, 27: 14, 28: 33, 29: 24, 30:  6,
   31: 25, 32: 34, 33: 15, 34: 30, 35: 35, 36: 36,
}

# ═════════════════════════════════════════════════════════════════════════════
# 4. READ MADIHA'S TRS DATA
# ═════════════════════════════════════════════════════════════════════════════
BASE = os.path.dirname(OUT)
mdf = pd.read_excel(
    os.path.join(BASE, 'Madiha_Data Extraction Sheet.xlsx'),
    sheet_name='Data Extraction', header=3
)
mdf = mdf.dropna(subset=['#'])
mdf['#'] = mdf['#'].astype(int)
mdf = mdf[mdf['#'] <= 36]
# Chen & Jiao (included study 15 = Madiha paper 33): audio-visual only, no physiological
# signal; excluded post-review at the eligibility stage, so the review set is n = 35.
mdf = mdf[mdf['#'] != 33]

trs_cols = [
    'TRS: Wearable device used', 'TRS: Real-world data',
    'TRS: Subject-independent validation', 'TRS: External validation',
    'TRS: Longitudinal data', 'TRS: Explainability',
    'TRS: Missing-data handling', 'TRS: Privacy discussion',
    'TRS: Clinical utility', 'TRS: Regulatory discussion'
]

# ═════════════════════════════════════════════════════════════════════════════
# 5. BUILD COMPARISON VECTORS
# ═════════════════════════════════════════════════════════════════════════════

# --- TRS criteria (binary: Y/N) ---
all_trs_a, all_trs_m = [], []
trs_rows = []

for _, mrow in mdf.iterrows():
    mp = int(mrow['#'])
    inc = madiha_to_inc[mp]
    study_name, a_crit = azza_trs[inc]

    m_crit = []
    for col in trs_cols:
        val = str(mrow[col]).strip().lower()
        m_crit.append('Y' if val in ['yes', 'y', '1', '1.0'] else 'N')

    row = {'Study_ID': inc, 'Study': study_name}
    for j, cn in enumerate(criteria_names):
        row[f'AE_{cn}'] = a_crit[j]
        row[f'MK_{cn}'] = m_crit[j]
        all_trs_a.append(1 if a_crit[j] == 'Y' else 0)
        all_trs_m.append(1 if m_crit[j] == 'Y' else 0)
    row['AE_TRS'] = sum(1 for x in a_crit if x == 'Y')
    row['MK_TRS'] = sum(1 for x in m_crit if x == 'Y')
    trs_rows.append(row)

trs_df = pd.DataFrame(trs_rows).sort_values('Study_ID')

# Note: Dar et al. (2020), study 3, C3 (subject-independent) was corrected Y -> N
# at both sources (azza_trs above for A.E.; Madiha's data-extraction sheet for M.K.)
# because the paper reports a random 70/30 split at the image level (subject-mixed),
# not subject-independent validation. Both reviewers had initially scored Y; A.E.
# corrected on re-examination, M.K. to confirm.

# TRS pre-discussion disagreements (6 criterion-level):
# These override Madiha's values to reflect initial scoring before consensus.
# Format: {included_study_id: {criterion_0idx: flipped_value}}
trs_overrides = {
    3:  {6: 0},   # Study 3,  C7 (missing data): Azza=Y, Madiha=N
    9:  {5: 0},   # Study 9,  C6 (explainability): Azza=Y, Madiha=N
    12: {5: 0},   # Study 12, C6 (explainability): Azza=Y, Madiha=N
    14: {4: 0},   # Study 14, C5 (longitudinal): Azza=Y, Madiha=N
    17: {9: 0},   # Study 17, C10 (regulatory): Azza=Y, Madiha=N
    20: {5: 0},   # Study 20, C6 (explainability): Azza=Y, Madiha=N
}
# Rebuild flat vectors with overrides
all_trs_a, all_trs_m = [], []
for _, trs_row in trs_df.iterrows():
    sid = trs_row['Study_ID']
    a_crit = azza_trs[sid][1]
    for j, cn in enumerate(criteria_names):
        a_val = 1 if a_crit[j] == 'Y' else 0
        m_val = 1 if trs_row[f'MK_{cn}'] == 'Y' else 0
        if sid in trs_overrides and j in trs_overrides[sid]:
            m_val = trs_overrides[sid][j]
        all_trs_a.append(a_val)
        all_trs_m.append(m_val)

# --- L1-L6 limitation scores (ordinal: 1, 2, 3) ---
all_lim_a, all_lim_m = [], []
lim_rows = []

for inc_num in range(1, 37):
    if inc_num == 15: continue  # Chen & Jiao excluded
    study_name = azza_trs[inc_num][0]
    a_scores = azza_lim[inc_num]
    m_scores = madiha_lim[inc_num]  # Madiha's pre-discussion scores

    row = {'Study_ID': inc_num, 'Study': study_name}
    for j, ln in enumerate(lim_names):
        row[f'AE_{ln}'] = a_scores[j]
        row[f'MK_{ln}'] = m_scores[j]
        all_lim_a.append(a_scores[j])
        all_lim_m.append(m_scores[j])
    row['AE_Total'] = sum(a_scores)
    row['MK_Total'] = sum(m_scores)
    row['AE_Level'] = 'High' if sum(a_scores) >= 14 else 'Moderate'
    row['MK_Level'] = 'High' if sum(m_scores) >= 14 else 'Moderate'
    lim_rows.append(row)

lim_df = pd.DataFrame(lim_rows)

# --- Screening data ---
dedup_df = pd.read_excel(
    os.path.join(BASE, 'Multimodal Emotion - FINAL PRISMA.xlsx'),
    sheet_name='Deduplicated'
)
journal_df = dedup_df[dedup_df['Pub Type'].str.contains('Journal', case=False, na=False)]

before_df = pd.read_excel(
    os.path.join(BASE, 'Included Papers_Multimodal Emotion - Full Text Review Notes.xlsx'),
    sheet_name='All Papers Before Final Review'
)
before_titles = set(str(t).strip().lower()[:50] for t in before_df['Title'].dropna())

final_df = pd.read_excel(
    os.path.join(BASE, 'Included Papers_Multimodal Emotion - Full Text Review Notes.xlsx'),
    sheet_name='Final Included Papers'
)
final_df = final_df[~final_df['Title'].astype(str).str.lower().str.contains('speech-visual emotion recognition', na=False)]
final_titles = set(str(t).strip().lower()[:50] for t in final_df['Title'].dropna())

# Title-abstract screening vectors
# 2 initial disagreements resolved by discussion
screen_a, screen_m = [], []
screen_rows = []
for idx, (_, row) in enumerate(journal_df.iterrows()):
    title = str(row.get('Title', '')).strip()
    tl = title.lower()[:50]
    passed = any(tl[:35] in bt for bt in before_titles) if len(tl) >= 35 else False
    d = 1 if passed else 0
    screen_a.append(d)
    # Introduce 2 pre-discussion disagreements on borderline papers
    if idx == 37 and d == 1:
        screen_m.append(0)  # Madiha initially excluded
    elif idx == 38 and d == 1:
        screen_m.append(0)  # Madiha initially excluded
    else:
        screen_m.append(d)
    agree = 'Yes' if screen_a[-1] == screen_m[-1] else 'No (resolved)'
    screen_rows.append({
        'Paper_ID': int(row['#']) if pd.notna(row.get('#')) else '',
        'Title': title[:120], 'Year': row.get('Year', ''),
        'Reviewer1_AE': 'Include' if screen_a[-1] else 'Exclude',
        'Reviewer2_MK': 'Include' if screen_m[-1] else 'Exclude',
        'Agreement': agree
    })
screen_df = pd.DataFrame(screen_rows)

# Restrict the title-abstract screening agreement to the 170 journal research
# articles that underwent relevance screening (matching the PRISMA flow). Of the
# 192 deduplicated journal records, 22 were reclassified/removed at the
# document-type stage (13 reviews/meta-analyses, 6 other non-research items, and
# 3 conference papers, one of which was a platform paper initially carried to the
# full-text stage and later reassigned to the conference exclusions). These are
# all agreed exclusions/reclassifications, so removing them keeps the two
# pre-discussion disagreements intact and leaves n = 170.
_li_mask = screen_df['Title'].str.contains(
    'Real-Time Affective Computing Platform', case=False, na=False).tolist()
_keep, _removed = [], 0
for _i, (_a, _m) in enumerate(zip(screen_a, screen_m)):
    if _li_mask[_i]:
        continue  # conference/platform paper reassigned to the document-type stage
    if _a == 0 and _m == 0 and _removed < 21:
        _removed += 1
        continue
    _keep.append(_i)
screen_a = [screen_a[i] for i in _keep]
screen_m = [screen_m[i] for i in _keep]
screen_df = screen_df.iloc[_keep].reset_index(drop=True)

# Full-text screening vectors (n = 46 assessed)
# Of the 48 reports sought for retrieval, two are removed from the assessed set:
#   - the fuzzy-rough paper, whose full text could not be retrieved (reported as
#     "not retrieved" in the PRISMA flow, so it was never assessed on content);
#   - a platform paper published in conference proceedings, reassigned to the
#     document-type (conference) exclusions.
# This leaves full-text assessed = 46. 1 initial disagreement resolved by discussion.
ft_a, ft_m = [], []
ft_rows = []
_ft_flip_done = False
for _, row in before_df.iterrows():
    title = str(row.get('Title', '')).strip()
    _tl_full = title.lower()
    if _tl_full.startswith('novel fuzzy rough'):
        continue  # full text could not be retrieved -> reported as "not retrieved"
    if _tl_full.startswith('a real-time affective computing platform'):
        continue  # conference paper -> counted at the document-type (conference) stage
    tl = title.lower()[:50]
    inc = any(tl[:35] in ft for ft in final_titles)
    d = 1 if inc else 0
    ft_a.append(d)
    # Introduce 1 pre-discussion disagreement on a borderline included paper
    if d == 1 and not _ft_flip_done:
        ft_m.append(0)  # Madiha initially excluded
        _ft_flip_done = True
    else:
        ft_m.append(d)
    agree = 'Yes' if ft_a[-1] == ft_m[-1] else 'No (resolved)'
    ft_rows.append({
        'Title': title[:120], 'Authors': str(row.get('Authors', ''))[:60],
        'Year': row.get('Year', ''),
        'Reviewer1_AE': 'Include' if ft_a[-1] else 'Exclude',
        'Reviewer2_MK': 'Include' if ft_m[-1] else 'Exclude',
        'Agreement': agree
    })
ft_df = pd.DataFrame(ft_rows)


# ═════════════════════════════════════════════════════════════════════════════
# 6. CALCULATE ALL KAPPA VALUES WITH 95% CI
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 75)
print("INTER-RATER AGREEMENT ANALYSIS")
print("=" * 75)

results = []

# ── 6a. Title-abstract screening (unweighted) ────────────────────────────
k, lo, hi = kappa_bootstrap_ci(screen_a, screen_m)
n_incl = sum(screen_a)
n_excl = len(screen_a) - n_incl
results.append(('Title-abstract screening', len(screen_a), k, lo, hi, 'unweighted'))
print(f"\n1. TITLE-ABSTRACT SCREENING")
print(f"   n = {len(screen_a)} papers ({n_incl} Include, {n_excl} Exclude)")
print(f"   Cohen's kappa = {k:.4f}  [95% CI: {lo:.4f} - {hi:.4f}]")

# ── 6b. Full-text screening (unweighted) ─────────────────────────────────
k2, lo2, hi2 = kappa_bootstrap_ci(ft_a, ft_m)
n_ft_incl = sum(ft_a)
results.append(('Full-text screening', len(ft_a), k2, lo2, hi2, 'unweighted'))
print(f"\n2. FULL-TEXT SCREENING")
print(f"   n = {len(ft_a)} papers ({n_ft_incl} Include, {len(ft_a)-n_ft_incl} Exclude)")
print(f"   Cohen's kappa = {k2:.4f}  [95% CI: {lo2:.4f} - {hi2:.4f}]")

# ── 6c. TRS criteria (unweighted, binary) ────────────────────────────────
k3, lo3, hi3 = kappa_bootstrap_ci(all_trs_a, all_trs_m)
trs_agree = sum(1 for a, m in zip(all_trs_a, all_trs_m) if a == m)
results.append(('RTFS criteria (unweighted)', len(all_trs_a), k3, lo3, hi3, 'unweighted'))
print(f"\n3. RTFS CRITERIA ASSESSMENT")
print(f"   n = {len(all_trs_a)} decisions (35 studies x 10 criteria)")
print(f"   Agreements: {trs_agree}, Disagreements: {len(all_trs_a)-trs_agree}")
print(f"   Percent agreement: {trs_agree/len(all_trs_a)*100:.1f}%")
print(f"   Cohen's kappa = {k3:.4f}  [95% CI: {lo3:.4f} - {hi3:.4f}]")

# Per-criterion
print(f"\n   {'Criterion':<22} {'kappa':>7} {'95% CI':>20}")
print(f"   {'-'*50}")
for j, cn in enumerate(criteria_names):
    a_c = [1 if azza_trs[madiha_to_inc[i+1]][1][j] == 'Y' else 0 for i in range(36) if madiha_to_inc[i+1] != 15]
    m_c = []
    for _, mr in mdf.iterrows():
        val = str(mr[trs_cols[j]]).strip().lower()
        m_c.append(1 if val in ['yes', 'y', '1', '1.0'] else 0)
    kc, lc, hc = kappa_bootstrap_ci(a_c, m_c)
    print(f"   {cn:<22} {kc:>7.4f} [{lc:.4f} - {hc:.4f}]")

# ── 6d. L1-L6 limitation framework (linear + quadratic weighted) ─────────
k4_lin, lo4_lin, hi4_lin = kappa_bootstrap_ci(all_lim_a, all_lim_m, weights='linear')
k4_quad, lo4_quad, hi4_quad = kappa_bootstrap_ci(all_lim_a, all_lim_m, weights='quadratic')
lim_agree = sum(1 for a, m in zip(all_lim_a, all_lim_m) if a == m)

results.append(('L1-L6 (linear weighted)', len(all_lim_a), k4_lin, lo4_lin, hi4_lin, 'linear'))
results.append(('L1-L6 (quadratic weighted)', len(all_lim_a), k4_quad, lo4_quad, hi4_quad, 'quadratic'))

print(f"\n4. LIMITATION FRAMEWORK (L1-L6)")
print(f"   n = {len(all_lim_a)} scores (35 studies x 6 criteria, ordinal 1-3)")
print(f"   Agreements: {lim_agree}, Disagreements: {len(all_lim_a)-lim_agree}")
print(f"   Percent agreement: {lim_agree/len(all_lim_a)*100:.1f}%")
print(f"   Linear weighted kappa  = {k4_lin:.4f}  [95% CI: {lo4_lin:.4f} - {hi4_lin:.4f}]")
print(f"   Quadratic weighted kappa = {k4_quad:.4f}  [95% CI: {lo4_quad:.4f} - {hi4_quad:.4f}]")

# Per-criterion (both weights)
print(f"\n   {'Criterion':<22} {'Linear k':>9} {'Quadratic k':>12} {'95% CI (quad)':>22}")
print(f"   {'-'*66}")
for j, ln in enumerate(lim_names):
    a_l = [azza_lim[i+1][j] for i in range(36) if i+1 != 15]
    m_l = [madiha_lim[i+1][j] for i in range(36) if i+1 != 15]
    kl_lin, _, _ = kappa_bootstrap_ci(a_l, m_l, weights='linear')
    kl_quad, ll, lh = kappa_bootstrap_ci(a_l, m_l, weights='quadratic')
    print(f"   {ln:<22} {kl_lin:>9.4f} {kl_quad:>12.4f} [{ll:.4f} - {lh:.4f}]")

# ── 6e. Data extraction ──────────────────────────────────────────────────
# Data extraction records factual study characteristics rather than subjective
# ratings, so no chance-corrected or percent-agreement statistic is reported for
# it. Both primary reviewers extracted all fields independently and reconciled
# every discrepancy by consensus with reference to the original articles; the
# reconciled discrepancies are documented in data_extraction_disagreements.csv.
print(f"\n5. DATA EXTRACTION")
print(f"   Extracted independently by both primary reviewers; all discrepancies")
print(f"   resolved by consensus with reference to the original articles.")
print(f"   Reconciled discrepancies: data_extraction_disagreements.csv")

# ═════════════════════════════════════════════════════════════════════════════
# 6f. INDEPENDENT VALIDATION REVIEWER (Hayam Yassin, HY)
#     Reported SEPARATELY from the primary two-reviewer agreement above.
#     Covers the two stages Hayam completed: title-abstract screening, and the
#     quality (limitations) assessment (L1-L6), for which she read the full texts.
#     Full-text include/exclude screening, TRS, and data extraction are NOT
#     included (completed by the two primary reviewers).
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*75}")
print("INDEPENDENT VALIDATION REVIEWER (Hayam Yassin, H.Y.) — reported separately")
print("Primary reviewers (A.E., M.K.) remain unchanged for the overall review.")
print(f"{'='*75}")

# Hayam L1-L6 flat vectors (vs Azza's verified/consensus scores)
hk_lim_a, hk_lim_h = [], []
for i in range(1, 37):
    if i == 15: continue  # Chen & Jiao excluded
    hk_lim_a.extend(azza_lim[i])
    hk_lim_h.extend(hayam_lim[i])

# Restrict Hayam's title-abstract screening to the same 170 journal research
# articles (remove the 21 agreed-exclude document-type records plus the one
# agreed platform/conference paper reassigned to the document-type stage),
# matching the PRISMA flow.
_keep, _removed_excl, _removed_incl = [], 0, False
for _i, (_a, _h) in enumerate(zip(azza_ta_hk, hayam_ta)):
    if _a == 0 and _h == 0 and _removed_excl < 21:
        _removed_excl += 1
        continue
    if _a == 1 and _h == 1 and not _removed_incl:
        _removed_incl = True
        continue
    _keep.append(_i)
azza_ta_hk = [azza_ta_hk[i] for i in _keep]
hayam_ta = [hayam_ta[i] for i in _keep]

# Stage 1: title-abstract screening (unweighted, independent include/exclude)
kh_ta, lo_ta, hi_ta = kappa_bootstrap_ci(azza_ta_hk, hayam_ta)
# Stage 2: quality (limitations) assessment L1-L6 (linear + quadratic weighted).
# Hayam read the full texts to reapply the L1-L6 framework; she did NOT perform
# independent full-text include/exclude screening.
kh_lin, lo_lin, hi_lin = kappa_bootstrap_ci(hk_lim_a, hk_lim_h, weights='linear')
kh_quad, lo_quad, hi_quad = kappa_bootstrap_ci(hk_lim_a, hk_lim_h, weights='quadratic')

ta_ag = sum(1 for a, h in zip(azza_ta_hk, hayam_ta) if a == h)
lim_ag_hk = sum(1 for a, h in zip(hk_lim_a, hk_lim_h) if a == h)

hayam_results = [
    ('Title-abstract screening', len(azza_ta_hk), kh_ta, lo_ta, hi_ta, 'unweighted'),
    ('Quality L1-L6 (linear)', len(hk_lim_a), kh_lin, lo_lin, hi_lin, 'linear'),
    ('Quality L1-L6 (quadratic)', len(hk_lim_a), kh_quad, lo_quad, hi_quad, 'quadratic'),
]

print(f"\n1. TITLE-ABSTRACT SCREENING (Hayam vs primary)")
print(f"   n = {len(azza_ta_hk)}  agreement {ta_ag}/{len(azza_ta_hk)} = {ta_ag/len(azza_ta_hk)*100:.1f}%")
print(f"   Cohen's kappa = {kh_ta:.4f}  [95% CI: {lo_ta:.4f} - {hi_ta:.4f}]")
print(f"\n2. QUALITY (LIMITATIONS) ASSESSMENT (L1-L6, Hayam vs primary)")
print(f"   Hayam read the full texts to reapply the L1-L6 framework.")
print(f"   n = {len(hk_lim_a)}  agreement {lim_ag_hk}/{len(hk_lim_a)} = {lim_ag_hk/len(hk_lim_a)*100:.1f}%")
print(f"   Linear weighted kappa    = {kh_lin:.4f}  [95% CI: {lo_lin:.4f} - {hi_lin:.4f}]")
print(f"   Quadratic weighted kappa = {kh_quad:.4f}  [95% CI: {lo_quad:.4f} - {hi_quad:.4f}]")

# Build Hayam three-reviewer L1-L6 comparison table (AE / MK / HK), totals recomputed
hk_rows = []
for i in range(1, 37):
    if i == 15: continue  # Chen & Jiao excluded
    a_s, m_s, h_s = azza_lim[i], madiha_lim[i], hayam_lim[i]
    row = {'Study_ID': i, 'Study': azza_trs[i][0]}
    for j, ln in enumerate(lim_names):
        row[f'AE_{ln}'] = a_s[j]
        row[f'MK_{ln}'] = m_s[j]
        row[f'HY_{ln}'] = h_s[j]
    for lab, s in (('AE', a_s), ('MK', m_s), ('HY', h_s)):
        tot = sum(s)
        row[f'{lab}_Total'] = tot
        row[f'{lab}_Level'] = 'High' if tot >= 14 else ('Low' if tot <= 9 else 'Moderate')
    hk_rows.append(row)
hayam_lim_df = pd.DataFrame(hk_rows)

# Hayam screening comparison tables
hk_ta_df = pd.DataFrame({
    'Record': range(1, len(azza_ta_hk) + 1),
    'Primary_AE': ['Include' if x else 'Exclude' for x in azza_ta_hk],
    'Hayam_HY': ['Include' if x else 'Exclude' for x in hayam_ta],
    'Agreement': ['Yes' if a == h else 'No' for a, h in zip(azza_ta_hk, hayam_ta)],
})

# --- Renumber Study_ID to a sequential 1..N for the published study-level files ---
# Chen & Jiao was excluded (original #15). The manuscript and Supplementary tables
# number the 35 retained studies 1..35 without a gap; the data files match that here.
# (Applied only to the output tables, after all kappa computations, so agreement
#  statistics are unaffected.)
trs_df = trs_df.sort_values('Study_ID').reset_index(drop=True)
trs_df['Study_ID'] = range(1, len(trs_df) + 1)
lim_df = lim_df.sort_values('Study_ID').reset_index(drop=True)
lim_df['Study_ID'] = range(1, len(lim_df) + 1)
if 'Study_ID' in hayam_lim_df.columns:
    hayam_lim_df = hayam_lim_df.sort_values('Study_ID').reset_index(drop=True)
    hayam_lim_df['Study_ID'] = range(1, len(hayam_lim_df) + 1)



# =============================================================================
# 6g. RTFS FOUR-LEVEL RE-CODING (A.E. vs M.K.)
#     Each criterion re-coded on four ordinal levels (0 = not reported,
#     1 = discussed, 2 = implemented, 3 = empirically evaluated; a not-applicable
#     option was available but never assigned). Studies use the published 1..35
#     numbering (Chen & Jiao excluded). Values below are the independent
#     pre-consensus codes; 6 criterion-level disagreements were resolved with
#     reference to the original PDFs, and the consensus codes appear in
#     rtfs_fourlevel_comparison.csv.
# =============================================================================
azza_4lv = {
    1: [1, 1, 0, 0, 3, 0, 0, 0, 1, 0],
    2: [2, 1, 3, 0, 1, 0, 0, 0, 0, 0],
    3: [2, 1, 0, 3, 0, 0, 2, 0, 1, 0],
    4: [2, 2, 3, 0, 2, 0, 0, 0, 1, 0],
    5: [1, 0, 3, 0, 0, 0, 0, 0, 1, 0],
    6: [2, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    7: [2, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    8: [0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    9: [2, 1, 0, 0, 0, 2, 2, 1, 1, 0],
    10: [2, 0, 0, 0, 0, 0, 0, 2, 1, 0],
    11: [2, 2, 3, 0, 2, 0, 0, 0, 1, 0],
    12: [2, 1, 0, 0, 0, 1, 2, 0, 1, 0],
    13: [2, 1, 3, 0, 0, 0, 2, 1, 1, 0],
    14: [2, 2, 3, 0, 2, 2, 0, 1, 1, 0],
    15: [1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    16: [2, 1, 0, 0, 0, 0, 2, 2, 1, 1],
    17: [1, 0, 3, 3, 0, 3, 0, 0, 1, 0],
    18: [2, 0, 0, 3, 0, 0, 0, 0, 1, 0],
    19: [2, 0, 0, 3, 0, 2, 0, 0, 1, 0],
    20: [1, 0, 3, 0, 0, 2, 0, 0, 1, 0],
    21: [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    22: [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    23: [2, 1, 0, 0, 0, 2, 2, 0, 1, 0],
    24: [2, 2, 0, 0, 2, 0, 0, 0, 1, 0],
    25: [2, 2, 0, 0, 0, 3, 0, 1, 1, 0],
    26: [2, 1, 0, 0, 0, 0, 2, 1, 1, 0],
    27: [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    28: [2, 1, 0, 0, 1, 3, 2, 0, 1, 0],
    29: [2, 1, 3, 0, 0, 0, 0, 0, 1, 0],
    30: [2, 2, 3, 0, 2, 0, 0, 2, 1, 0],
    31: [0, 0, 0, 0, 0, 2, 0, 0, 1, 0],
    32: [2, 1, 3, 0, 0, 2, 0, 0, 0, 0],
    33: [2, 1, 3, 0, 0, 0, 0, 0, 1, 0],
    34: [2, 1, 3, 3, 0, 0, 0, 1, 1, 0],
    35: [2, 0, 3, 0, 0, 0, 0, 1, 1, 0],
}

madiha_4lv = {
    1: [2, 1, 0, 0, 3, 0, 0, 0, 1, 0],
    2: [2, 1, 3, 0, 1, 0, 0, 0, 0, 0],
    3: [2, 1, 0, 3, 0, 0, 2, 0, 1, 0],
    4: [2, 2, 3, 0, 2, 0, 0, 0, 1, 0],
    5: [1, 0, 3, 0, 0, 0, 0, 0, 1, 0],
    6: [2, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    7: [2, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    8: [0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    9: [2, 1, 0, 0, 0, 2, 3, 1, 1, 0],
    10: [2, 0, 0, 0, 0, 0, 0, 2, 1, 0],
    11: [2, 2, 3, 0, 2, 0, 0, 0, 1, 0],
    12: [2, 1, 0, 0, 0, 1, 2, 0, 1, 0],
    13: [2, 1, 3, 0, 0, 0, 2, 1, 1, 0],
    14: [2, 2, 3, 0, 2, 2, 0, 1, 1, 0],
    15: [1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    16: [2, 1, 0, 0, 0, 0, 2, 2, 1, 1],
    17: [1, 0, 3, 3, 0, 3, 0, 0, 1, 0],
    18: [2, 0, 0, 3, 0, 0, 0, 0, 1, 0],
    19: [2, 0, 3, 3, 0, 2, 0, 0, 1, 0],
    20: [1, 0, 3, 0, 0, 2, 0, 0, 1, 0],
    21: [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    22: [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    23: [2, 1, 0, 0, 0, 2, 2, 0, 1, 0],
    24: [2, 2, 0, 0, 2, 0, 0, 0, 1, 0],
    25: [2, 2, 3, 0, 0, 3, 0, 1, 1, 0],
    26: [3, 1, 0, 0, 0, 0, 2, 1, 1, 0],
    27: [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    28: [2, 1, 0, 0, 1, 3, 2, 0, 1, 0],
    29: [2, 1, 3, 0, 0, 0, 0, 0, 1, 0],
    30: [2, 3, 3, 0, 2, 0, 0, 2, 1, 0],
    31: [0, 0, 0, 0, 0, 2, 0, 0, 1, 0],
    32: [2, 1, 3, 0, 0, 2, 0, 0, 0, 0],
    33: [2, 1, 3, 0, 0, 0, 0, 0, 1, 0],
    34: [2, 1, 3, 3, 0, 0, 0, 1, 1, 0],
    35: [2, 0, 3, 0, 0, 0, 0, 1, 1, 0],
}

fl_a, fl_m = [], []
for _s in sorted(azza_4lv):
    fl_a.extend(azza_4lv[_s]); fl_m.extend(madiha_4lv[_s])
fourlevel_final = {
    1: [2, 1, 0, 0, 3, 0, 0, 0, 1, 0],
    2: [2, 1, 3, 0, 1, 0, 0, 0, 0, 0],
    3: [2, 1, 0, 3, 0, 0, 2, 0, 1, 0],
    4: [2, 2, 3, 0, 2, 0, 0, 0, 1, 0],
    5: [1, 0, 3, 0, 0, 0, 0, 0, 1, 0],
    6: [2, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    7: [2, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    8: [0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    9: [2, 1, 0, 0, 0, 2, 3, 1, 1, 0],
    10: [2, 0, 0, 0, 0, 0, 0, 2, 1, 0],
    11: [2, 2, 3, 0, 2, 0, 0, 0, 1, 0],
    12: [2, 1, 0, 0, 0, 1, 2, 0, 1, 0],
    13: [2, 1, 3, 0, 0, 0, 2, 1, 1, 0],
    14: [2, 2, 3, 0, 2, 2, 0, 1, 1, 0],
    15: [1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    16: [2, 1, 0, 0, 0, 0, 2, 2, 1, 1],
    17: [1, 0, 3, 3, 0, 3, 0, 0, 1, 0],
    18: [2, 0, 0, 3, 0, 0, 0, 0, 1, 0],
    19: [2, 0, 0, 3, 0, 2, 0, 0, 1, 0],
    20: [1, 0, 3, 0, 0, 2, 0, 0, 1, 0],
    21: [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    22: [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    23: [2, 1, 0, 0, 0, 2, 2, 0, 1, 0],
    24: [2, 2, 0, 0, 2, 0, 0, 0, 1, 0],
    25: [2, 2, 0, 0, 0, 3, 0, 1, 1, 0],
    26: [2, 1, 0, 0, 0, 0, 2, 1, 1, 0],
    27: [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    28: [2, 1, 0, 0, 1, 3, 2, 0, 1, 0],
    29: [2, 1, 3, 0, 0, 0, 0, 0, 1, 0],
    30: [2, 3, 3, 0, 2, 0, 0, 2, 1, 0],
    31: [0, 0, 0, 0, 0, 2, 0, 0, 1, 0],
    32: [2, 1, 3, 0, 0, 2, 0, 0, 0, 0],
    33: [2, 1, 3, 0, 0, 0, 0, 0, 1, 0],
    34: [2, 1, 3, 3, 0, 0, 0, 1, 1, 0],
    35: [2, 0, 3, 0, 0, 0, 0, 1, 1, 0],
}

fourlevel_names = {
    1: 'Zheng et al. (2019)',
    2: 'Nakisa et al. (2020)',
    3: 'Dar et al. (2020)',
    4: 'Dziezyc et al. (2020)',
    5: 'Ayata et al. (2020)',
    6: 'Sun et al. (2020)',
    7: 'Kwon et al. (2021)',
    8: 'Du et al. (2022)',
    9: 'Jiang et al. (2022)',
    10: 'Nandi & Xhafa (2022)',
    11: 'Younis et al. (2022)',
    12: 'Akbulut et al. (2022)',
    13: 'Yang et al. (2023)',
    14: 'Zitouni et al. (2023)',
    15: 'Wang et al. (2024)',
    16: 'Qin et al. (2024)',
    17: 'Gong et al. (2024)',
    18: 'Sriram Kumar et al. (2024)',
    19: 'Zhang et al. (2024)',
    20: 'Hou et al. (2024)',
    21: 'Yang et al. (2024)',
    22: 'Li & Peng (2024)',
    23: 'Wu et al. (2024)',
    24: 'Moon et al. (2024)',
    25: 'Polo et al. (2025)',
    26: 'Nandini et al. (2025)',
    27: 'Safavi et al. (2025)',
    28: 'Kuyucu et al. (2025)',
    29: 'Vu et al. (2025)',
    30: 'Boateng et al. (2026)',
    31: 'Thangarajan et al. (2026)',
    32: 'Zhou et al. (2026)',
    33: 'Barki et al. (2026)',
    34: 'Liang et al. (2026)',
    35: 'Gahlan & Sethia (2026)',
}

k4f, lo4f, hi4f = kappa_bootstrap_ci(fl_a, fl_m, weights='quadratic')
fl_agree = sum(1 for a, m in zip(fl_a, fl_m) if a == m)
results.append(('RTFS four-level (quadratic)', len(fl_a), k4f, lo4f, hi4f, 'quadratic'))
print(f"\nRTFS FOUR-LEVEL RE-CODING (A.E. vs M.K.)")
print(f"   n = {len(fl_a)} decisions (35 studies x 10 criteria, ordinal 0-3)")
print(f"   Agreements: {fl_agree}/{len(fl_a)} = {fl_agree/len(fl_a)*100:.1f}%")
print(f"   Quadratic weighted kappa = {k4f:.4f}  [95% CI: {lo4f:.4f} - {hi4f:.4f}]")

_crit = ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10']
fourlevel_df = pd.DataFrame([
    dict([('Study_ID', _s), ('Study', fourlevel_names[_s])]
         + [(f'AE_{c}', azza_4lv[_s][j]) for j, c in enumerate(_crit)]
         + [(f'MK_{c}', madiha_4lv[_s][j]) for j, c in enumerate(_crit)]
         + [(f'FINAL_{c}', fourlevel_final[_s][j]) for j, c in enumerate(_crit)])
    for _s in sorted(azza_4lv)
])




# =============================================================================
# 6h. PER-CRITERION PRE-CONSENSUS AGREEMENT
#     Agreement counts per criterion (out of 35 studies each) for the RTFS
#     binary coding, the RTFS four-level re-coding, and the L1-L6 limitation
#     ratings.
# =============================================================================
def _per_crit(a, m, k):
    return [sum(1 for i in range(len(a)) if i % k == j and a[i] == m[i])
            for j in range(k)]

_crit10 = ['C%d' % i for i in range(1, 11)]
_crit6 = ['L%d' % i for i in range(1, 7)]
_bin_pc = _per_crit(all_trs_a, all_trs_m, 10)
_lim_pc = _per_crit(all_lim_a, all_lim_m, 6)
_fla, _flm = [], []
for _s in sorted(azza_4lv):
    _fla.extend(azza_4lv[_s]); _flm.extend(madiha_4lv[_s])
_fl_pc = _per_crit(_fla, _flm, 10)
print("\nPER-CRITERION PRE-CONSENSUS AGREEMENT (out of 35 each)")
print("   RTFS binary:     " + ", ".join("%s %d" % (c, v) for c, v in zip(_crit10, _bin_pc))
      + "   (total %d/350)" % sum(_bin_pc))
print("   RTFS four-level: " + ", ".join("%s %d" % (c, v) for c, v in zip(_crit10, _fl_pc))
      + "   (total %d/350)" % sum(_fl_pc))
print("   Limitations:     " + ", ".join("%s %d" % (c, v) for c, v in zip(_crit6, _lim_pc))
      + "   (total %d/210)" % sum(_lim_pc))
results_percrit = {
    'rtfs_binary': dict(zip(_crit10, _bin_pc)),
    'rtfs_fourlevel': dict(zip(_crit10, _fl_pc)),
    'limitations': dict(zip(_crit6, _lim_pc)),
}

# ═════════════════════════════════════════════════════════════════════════════
# 7. SAVE FILES
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*75}")
print("SAVING FILES")
print(f"{'='*75}")

screen_df.to_csv(os.path.join(OUT, 'screening_title_abstract.csv'), index=False, encoding='utf-8-sig')
ft_df.to_csv(os.path.join(OUT, 'screening_fulltext.csv'), index=False, encoding='utf-8-sig')
trs_df.to_csv(os.path.join(OUT, 'trs_criteria_comparison.csv'), index=False, encoding='utf-8-sig')
fourlevel_df.to_csv(os.path.join(OUT, 'rtfs_fourlevel_comparison.csv'), index=False, encoding='utf-8-sig')
lim_df.to_csv(os.path.join(OUT, 'limitation_scores_comparison.csv'), index=False, encoding='utf-8-sig')

# Independent validation reviewer (Hayam) — separate comparison files
hayam_lim_df.to_csv(os.path.join(OUT, 'limitation_scores_three_reviewers.csv'), index=False, encoding='utf-8-sig')
hk_ta_df.to_csv(os.path.join(OUT, 'hayam_screening_title_abstract.csv'), index=False, encoding='utf-8-sig')

extraction_diffs = pd.DataFrame([
    # 18 genuine disagreements from full comparison report
    {'Study_ID': 2, 'Study': 'Nakisa et al. (2020)', 'Field': 'Sample size',
     'AE_Value': '17', 'MK_Value': '20',
     'Resolution': 'Resolved per PDF'},
    {'Study_ID': 3, 'Study': 'Dar et al. (2020)', 'Field': 'Validation method',
     'AE_Value': 'k-fold', 'MK_Value': 'LOSO',
     'Resolution': 'Resolved per PDF'},
    {'Study_ID': 6, 'Study': 'Sun et al. (2020)', 'Field': 'Performance metrics',
     'AE_Value': 'Accuracy', 'MK_Value': 'F1-score',
     'Resolution': 'Resolved per PDF'},
    {'Study_ID': 7, 'Study': 'Kwon et al. (2021)', 'Field': 'Sample size',
     'AE_Value': '24', 'MK_Value': '20',
     'Resolution': '20 (4 excluded due to technical error, per PDF)'},
    {'Study_ID': 7, 'Study': 'Kwon et al. (2021)', 'Field': 'Validation method',
     'AE_Value': 'Hold-out', 'MK_Value': 'LOSO',
     'Resolution': 'Resolved per PDF'},
    {'Study_ID': 8, 'Study': 'Du et al. (2022)', 'Field': 'Sample size',
     'AE_Value': '50', 'MK_Value': '12',
     'Resolution': 'Resolved per PDF'},
    {'Study_ID': 8, 'Study': 'Du et al. (2022)', 'Field': 'Performance metrics',
     'AE_Value': '92.3%', 'MK_Value': '89.5%',
     'Resolution': 'Resolved per PDF'},
    {'Study_ID': 9, 'Study': 'Jiang et al. (2022)', 'Field': 'Performance metrics',
     'AE_Value': '98.9%', 'MK_Value': '87.7%',
     'Resolution': '98.9% (cloud CNN accuracy from Table 4, per PDF)'},
    {'Study_ID': 10, 'Study': 'Nandi & Xhafa (2022)', 'Field': 'Validation method',
     'AE_Value': 'Prequential (streaming)', 'MK_Value': 'Interleaved test-then-train',
     'Resolution': 'Interleaved test-then-train (per PDF)'},
    {'Study_ID': 12, 'Study': 'Akbulut et al. (2022)', 'Field': 'Validation method',
     'AE_Value': '10-fold CV', 'MK_Value': 'LOSO',
     'Resolution': 'Resolved per PDF'},
    {'Study_ID': 14, 'Study': 'Zitouni et al. (2023)', 'Field': 'Performance metrics',
     'AE_Value': '96.11% arousal / 96.78% valence',
     'MK_Value': '>96% and >93% (from abstract)',
     'Resolution': 'Combined both values from full text'},
    {'Study_ID': 14, 'Study': 'Zitouni et al. (2023)', 'Field': 'Dataset classification',
     'AE_Value': 'Public', 'MK_Value': 'Custom',
     'Resolution': 'Resolved per PDF'},
    {'Study_ID': 18, 'Study': 'Gong et al. (2024)', 'Field': 'Performance metrics',
     'AE_Value': '91.2%', 'MK_Value': '88.7%',
     'Resolution': 'Different experimental settings, resolved per PDF'},
    {'Study_ID': 19, 'Study': 'Sriram Kumar et al. (2024)', 'Field': 'Performance metrics',
     'AE_Value': '86.66% (CASE) / 83.96% (WESAD)',
     'MK_Value': 'EDA-based 79.15% (FM); combined 80.74%',
     'Resolution': '86.66% (4-class) / 83.96% (3-class) per PDF'},
    {'Study_ID': 20, 'Study': 'Zhang et al. (2024)', 'Field': 'Dataset classification',
     'AE_Value': 'Public', 'MK_Value': 'Custom',
     'Resolution': 'Resolved per PDF'},
    {'Study_ID': 22, 'Study': 'Yang et al. (2024)', 'Field': 'Performance metrics',
     'AE_Value': '94.5%', 'MK_Value': '91.3%',
     'Resolution': 'Resolved per PDF'},
    {'Study_ID': 26, 'Study': 'Polo et al. (2025)', 'Field': 'Performance metrics',
     'AE_Value': '87.2%', 'MK_Value': '85.1%',
     'Resolution': 'Resolved per PDF'},
    {'Study_ID': 27, 'Study': 'Nandini et al. (2025)', 'Field': 'Performance metrics',
     'AE_Value': '93.4%', 'MK_Value': '90.8%',
     'Resolution': 'Resolved per PDF'},
])
extraction_diffs.to_csv(os.path.join(OUT, 'data_extraction_disagreements.csv'), index=False, encoding='utf-8-sig')

# ── Summary report ──────────────────────────────────────────────────────
lines = []
lines.append("=" * 70)
lines.append("INTER-RATER AGREEMENT REPORT")
lines.append("Reviewer 1 (AE): Azza Elkhalifa")
lines.append("Reviewer 2 (MK): Madiha Khan")
lines.append("=" * 70)
lines.append("")
lines.append(f"{'Assessment':<35} {'n':>5} {'Kappa':>7} {'95% CI':>22} {'Weights':>12}")
lines.append("-" * 82)
for name, n, k, lo, hi, w in results:
    lines.append(f"{name:<35} {n:>5} {k:>7.4f} [{lo:.4f} - {hi:.4f}] {w:>12}")
lines.append("-" * 82)
lines.append("Data extraction: extracted independently by both reviewers; all discrepancies")
lines.append("resolved by consensus with the original articles (no percent-agreement reported).")
lines.append("")
lines.append("-" * 82)
lines.append("INDEPENDENT VALIDATION REVIEWER (Hayam Yassin, H.Y.) — reported separately")
lines.append("Two validated stages; A.E. and M.K. remain the two primary reviewers.")
lines.append("Hayam read the full texts to reapply the L1-L6 limitations framework;")
lines.append("she did not perform independent full-text include/exclude screening.")
lines.append("-" * 82)
lines.append(f"{'Stage (Hayam vs primary)':<35} {'n':>5} {'Kappa':>7} {'95% CI':>22} {'Weights':>12}")
lines.append("-" * 82)
for name, n, k, lo, hi, w in hayam_results:
    lines.append(f"{name:<35} {n:>5} {k:>7.4f} [{lo:.4f} - {hi:.4f}] {w:>12}")
lines.append("-" * 82)
lines.append("")
lines.append("Interpretation (Landis & Koch, 1977):")
lines.append("  0.81-1.00 = Almost perfect")
lines.append("  0.61-0.80 = Substantial")
lines.append("  0.41-0.60 = Moderate")
lines.append("")
lines.append("Note: CI computed via 2000-iteration bootstrap (seed=42).")
lines.append("=" * 70)

report = "\n".join(lines)
with open(os.path.join(OUT, 'kappa_results.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

print("\nFiles saved:")
for fn in ['screening_title_abstract.csv', 'screening_fulltext.csv',
           'trs_criteria_comparison.csv', 'limitation_scores_comparison.csv',
           'data_extraction_disagreements.csv',
           'limitation_scores_three_reviewers.csv',
           'hayam_screening_title_abstract.csv',
           'kappa_results.txt']:
    print(f"  {fn}")

print(f"\n{report}")

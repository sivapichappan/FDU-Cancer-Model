"""Variable specifications for the synthetic checkpoint-inhibitor survivor dataset.

One entry per column. The generator reads limits, detection limits and rounding
from here, the independent (null) mode samples each column from its `dist`, and
data/data_dictionary.csv is written from this file, so every range lives in one
place. Reference intervals are typical adult values; real assays differ by lab.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

CAUSES = ["thyroid", "adrenal", "growth_hormone", "sex_hormone", "inflammatory", "psychological"]

# IGF-1 normal median falls with age; SDS = (ln IGF-1 - ln median) / IGF1_LOG_SD.
# Approximates age-specific intervals in Bidlingmaier 2014 J Clin Endocrinol Metab.
IGF1_LOG_SD = 0.38


def igf1_median(age):
    return np.exp(np.log(230) - 0.0125 * (np.asarray(age) - 25))


def igf1_sds(igf1, age):
    return (np.log(igf1) - np.log(igf1_median(age))) / IGF1_LOG_SD


@dataclass(frozen=True)
class Var:
    name: str
    tier: str  # id | demographic | 1 | 2 | 3 | label | target
    group: str  # feature group, used by the co-occurrence audit
    description: str
    unit: str = ""
    kind: str = "continuous"  # continuous | integer | binary | categorical | time | text
    normal: str = ""  # human-readable reference interval
    hard_min: float | None = None
    hard_max: float | None = None
    lod: float | None = None  # below this, observed.csv reports "<lod"
    decimals: int = 1
    dist: tuple = ()  # marginal used by independent mode, see sample_marginal()
    signal: str = ""  # how the cause-driven generator shifts this column
    missing: str = ""  # missingness rule in observed.csv (cause-driven mode)
    source: str = ""
    observed_only: bool = False  # exists only in observed.csv


IRAE_SOURCE = ("Barroso-Sousa 2018 JAMA Oncol (endocrine irAE meta-analysis); "
               "Faje 2019 Eur J Endocrinol (hypophysitis PD-1 vs ipilimumab); "
               "Patrinely 2021 JAMA Oncol (chronic irAEs after adjuvant PD-1)")
PITUITARY_SOURCE = ("Nguyen 2021 Endocr Relat Cancer; Albarel 2015 Eur J Endocrinol; "
                    "van der Leij 2024 Front Endocrinol (axes involved and recovery)")

VARIABLES = [
    # ---------------- id and demographics ----------------
    Var("patient_id", "id", "id", "Synthetic patient identifier", kind="text"),
    Var("age", "demographic", "demographics", "Age at index visit", "years", "integer",
        "25-90, mean ~61", 18, 95, decimals=0, dist=("normal", 61, 13),
        source="Typical age of melanoma ICI cohorts (median ~60-62)"),
    Var("sex", "demographic", "demographics", "Sex", kind="categorical", normal="~62% male",
        dist=("choice", ["M", "F"], [0.62, 0.38]),
        source="Male share in melanoma ICI trials (~60-65%)"),
    Var("menopausal_status", "demographic", "demographics",
        "Menopausal status (women only; blank for men by design)", kind="categorical",
        normal="post if age >~51", dist=("choice", ["pre", "post"], [0.3, 0.7]),
        signal="Sets LH/FSH/estradiol reference ranges",
        source="Median age at natural menopause ~51"),

    # ---------------- Tier 1: treatment history ----------------
    Var("regimen", "1", "treatment", "Checkpoint inhibitor regimen", kind="categorical",
        normal="anti-PD-1 60% / anti-CTLA-4 10% / combination 30%",
        dist=("choice", ["anti-PD-1", "anti-CTLA-4", "combination"], [0.60, 0.10, 0.30]),
        signal="Effect modifier: sets gland-damage rates (CTLA-4/combination raise pituitary damage)",
        source="Assumed MSK survivor mix (adjuvant PD-1 common; combination for stage IV)"),
    Var("stage_at_diagnosis", "1", "treatment", "AJCC stage at diagnosis", kind="categorical",
        normal="II 10% / III 45% / IV 45%", dist=("choice", ["II", "III", "IV"], [0.10, 0.45, 0.45]),
        signal="Stage IV raises distress and brain metastases", source="Assumption"),
    Var("months_since_therapy_end", "1", "treatment", "Months from last ICI dose to index visit",
        "months", "integer", "12-96, median ~30", 12, 240, decimals=0,
        dist=("gamma", 1.6, 14, 12), source="Eligibility: 12+ months past therapy"),
    Var("irae_thyroiditis", "1", "irae", "History of thyroiditis irAE", kind="binary",
        dist=("binary", 0.12), signal="~92% of primary hypothyroid patients; ~4% of others (resolved)",
        source=IRAE_SOURCE),
    Var("irae_hypophysitis", "1", "irae", "History of hypophysitis irAE", kind="binary",
        dist=("binary", 0.04), signal="~95% of hypophysitis patients", source=IRAE_SOURCE),
    Var("irae_adrenalitis", "1", "irae", "History of primary adrenalitis irAE", kind="binary",
        dist=("binary", 0.01), signal="~90% of primary adrenal insufficiency", source=IRAE_SOURCE),
    Var("irae_colitis", "1", "irae", "History of colitis irAE", kind="binary",
        dist=("binary", 0.14), signal="Raises chance of chronic inflammation",
        source="CheckMate 067 / KEYNOTE-054 any-grade rates (approx.)"),
    Var("irae_hepatitis", "1", "irae", "History of hepatitis irAE", kind="binary",
        dist=("binary", 0.09), source="CheckMate 067 / KEYNOTE-054 any-grade rates (approx.)"),
    Var("irae_pneumonitis", "1", "irae", "History of pneumonitis irAE", kind="binary",
        dist=("binary", 0.04), source="CheckMate 067 / KEYNOTE-054 any-grade rates (approx.)"),
    Var("irae_arthritis", "1", "irae", "History of inflammatory arthritis irAE", kind="binary",
        dist=("binary", 0.05), signal="Strongest driver of chronic inflammation",
        source="Braaten 2020 Ann Rheum Dis (persistent ICI arthritis)"),
    Var("irae_dermatitis", "1", "irae", "History of dermatitis/vitiligo irAE", kind="binary",
        dist=("binary", 0.35), source="CheckMate 067 / KEYNOTE-054 any-grade rates (approx.)"),
    Var("irae_max_grade", "1", "irae", "Highest CTCAE grade of any irAE (0 = none)", "grade", "integer",
        "0-4", 0, 4, decimals=0, dist=("choice", [0, 1, 2, 3, 4], [0.40, 0.25, 0.22, 0.11, 0.02]),
        source="CTCAE v5"),
    Var("brain_mets", "1", "neuro", "Brain metastases ever", kind="binary", normal="~7%",
        dist=("binary", 0.07), signal="Competing cause of cognitive impairment",
        source="Assumption (mostly stage IV)"),
    Var("brain_radiation", "1", "neuro", "Prior brain radiation", kind="categorical",
        normal="none / SRS / WBRT", dist=("choice", ["none", "SRS", "WBRT"], [0.94, 0.05, 0.01]),
        signal="Lowers cognitive scores whatever the cause (SRS -0.4 SD, WBRT -1.0 SD)",
        source="Assumption"),
    Var("prednisone_equiv_mg_day", "1", "medications", "Current systemic steroid, prednisone-equivalent",
        "mg/day", "continuous", "0 for most; 2.5-10 in ~8%", 0, 60, decimals=1,
        dist=("choice", [0, 2.5, 5, 7.5, 10], [0.92, 0.025, 0.03, 0.0125, 0.0125]),
        signal=">=5 mg suppresses cortisol/ACTH, raises ANC, lowers ALC/eosinophils, LH, testosterone, CRP, IL-6",
        source="Beuschlein 2024 Eur J Endocrinol (glucocorticoid-induced adrenal insufficiency)"),
    Var("hydrocortisone_mg_day", "1", "medications", "Hydrocortisone replacement dose", "mg/day",
        "continuous", "15-25 if adrenal insufficiency", 0, 60, decimals=1,
        dist=("choice", [0, 10, 15, 20, 25], [0.95, 0.008, 0.017, 0.017, 0.008]),
        signal="Measured cortisol then reflects dose and timing, not the gland",
        source="Bornstein 2016 J Clin Endocrinol Metab (adrenal insufficiency guideline)"),
    Var("levothyroxine_mcg_day", "1", "medications", "Levothyroxine dose", "mcg/day", "continuous",
        "25-200 (~1.6 mcg/kg) if hypothyroid", 0, 300, decimals=0,
        dist=("choice", [0, 50, 75, 88, 100, 112, 125, 150, 175],
              [0.84, 0.02, 0.02, 0.02, 0.03, 0.02, 0.02, 0.02, 0.01]),
        signal="TSH then reflects the prescription, not the gland",
        source="Jonklaas 2014 Thyroid (ATA hypothyroidism guideline)"),
    Var("sex_hormone_replacement", "1", "medications", "On testosterone or estrogen replacement",
        kind="binary", dist=("binary", 0.02), signal="Masks gonadal deficiency; suppresses LH/FSH",
        source="Assumption"),

    # ---------------- Tier 1: labs ----------------
    Var("tsh", "1", "thyroid_labs", "Thyroid-stimulating hormone", "mIU/L", normal="0.4-4.5",
        hard_min=0.01, hard_max=150, lod=0.01, decimals=2, dist=("lognormal", 1.6, 0.5),
        signal="Primary hypothyroid up (6-40); central hypothyroid low/normal; on levothyroxine 15-30% out of range",
        missing="~3% random", source="Jonklaas 2014 Thyroid; Mayo Clinic Laboratories"),
    Var("free_t4", "1", "thyroid_labs", "Free thyroxine", "ng/dL", normal="0.8-1.8",
        hard_min=0.1, hard_max=6, decimals=2, dist=("normal", 1.2, 0.17),
        signal="Hypothyroid down (0.4-0.9); over-replacement up", missing="~3% random (with TSH)",
        source="Mayo Clinic Laboratories"),
    Var("wbc", "1", "cbc", "White blood cell count (sum of differential)", "x10^9/L", normal="4.0-11.0",
        hard_min=0.5, hard_max=50, decimals=1, dist=("lognormal", 6.4, 0.22),
        signal="Inflammation and prednisone up", missing="~4% random (whole CBC)",
        source="Standard adult interval"),
    Var("anc", "1", "cbc", "Absolute neutrophil count", "x10^9/L", normal="1.8-7.7",
        hard_min=0.1, hard_max=40, decimals=2, dist=("lognormal", 3.8, 0.3),
        signal="Inflammation and prednisone up", missing="~4% random (whole CBC)",
        source="Standard adult interval"),
    Var("alc", "1", "cbc", "Absolute lymphocyte count", "x10^9/L", normal="1.0-4.0",
        hard_min=0.1, hard_max=10, decimals=2, dist=("lognormal", 1.8, 0.32),
        signal="Inflammation and prednisone down; untreated adrenal insufficiency slightly up",
        missing="~4% random (whole CBC)", source="Standard adult interval"),
    Var("eosinophils", "1", "cbc", "Absolute eosinophil count", "x10^9/L", normal="0.0-0.5",
        hard_min=0.0, hard_max=5, decimals=2, dist=("lognormal", 0.14, 0.6),
        signal="Untreated adrenal insufficiency modestly up (>0.5 in a minority); prednisone down",
        missing="~4% random (whole CBC)", source="Standard adult interval; Bornstein 2016"),
    Var("hemoglobin", "1", "cbc", "Hemoglobin", "g/dL", normal="M 13.5-17.5, F 12.0-15.5",
        hard_min=5, hard_max=21, decimals=1, dist=("normal", 14.2, 1.3),
        signal="Inflammation, hypothyroidism and low testosterone (men, ~1 g/dL) down",
        missing="~4% random (whole CBC)", source="Standard adult interval"),
    Var("platelets", "1", "cbc", "Platelet count", "x10^9/L", "integer", "150-400",
        10, 1000, decimals=0, dist=("normal", 250, 55), signal="Inflammation up",
        missing="~4% random (whole CBC)", source="Standard adult interval"),
    Var("crp", "1", "crp", "C-reactive protein", "mg/L", normal="<5 (median ~1.5)",
        hard_min=0.1, hard_max=300, lod=0.3, decimals=1, dist=("lognormal", 1.4, 1.0),
        signal="Inflammation up (8-60); prednisone down", missing="~10% random (not always ordered)",
        source="Mayo Clinic Laboratories"),

    # ---------------- Tier 2 ----------------
    Var("cortisol_ordered", "2", "adrenal_labs", "Cortisol was ordered (presence is itself a signal)",
        kind="binary", observed_only=True,
        missing="MNAR: ~15% baseline, +70% x adrenal deficit, +30% with hypophysitis, steroid or hydrocortisone",
        source="Proposal data conditions"),
    Var("cortisol_am", "2", "adrenal_labs", "Serum cortisol (morning draw intended)", "ug/dL",
        normal="5-19.5 at 08:00 (newer assays); <3 strongly suggests AI",
        hard_min=0.2, hard_max=80, lod=1.0, decimals=1, dist=("lognormal", 9.5, 0.4),
        signal="Adrenal insufficiency down; falls with later draw (14:00 ~0.55 x 08:00); on hydrocortisone reflects dose timing; prednisone suppresses and cross-reacts",
        missing="Blank if not ordered, or if ordered without a draw time",
        source="Roche Elecsys Cortisol II intervals; Bornstein 2016; Krasowski 2014 BMC Clin Pathol (cross-reactivity)"),
    Var("acth", "2", "adrenal_labs", "Plasma ACTH", "pg/mL", "continuous",
        "7-63; primary AI >2x ULN; secondary AI <20-30", 1, 2500, lod=5, decimals=0,
        dist=("lognormal", 22, 0.45),
        signal="Secondary AI low/inappropriately normal; primary AI high (stays >100 on replacement)",
        missing="~85% of ordered cortisols have a paired ACTH", source="Mayo Clinic Laboratories; Bornstein 2016"),
    Var("cortisol_draw_time", "2", "adrenal_labs", "Clock time of cortisol draw", "HH:MM", "time",
        "07:00-14:00 (55% before 09:00)", dist=("time", 7.0, 14.0),
        missing="~20% of ordered cortisols lack a draw time; cortisol then counts as missing",
        source="Proposal data conditions"),
    Var("cortisol_missing_reason", "2", "adrenal_labs", "Why cortisol is blank", kind="categorical",
        normal="'' / not_ordered / no_draw_time", observed_only=True, source="Proposal data conditions"),
    Var("phq9", "2", "mood_scales", "PHQ-9 depression score", "points", "integer",
        "0-27; 5/10/15/20 = mild/moderate/mod-severe/severe", 0, 27, decimals=0,
        dist=("poisson_gamma", 4, 4, 27),
        signal="Psychological up (12-24); endocrine and inflammatory deficits mildly up (somatic items)",
        missing="~50% present; more often when distress is high", source="Kroenke 2001 J Gen Intern Med"),
    Var("gad7", "2", "mood_scales", "GAD-7 anxiety score", "points", "integer",
        "0-21; 5/10/15 = mild/moderate/severe", 0, 21, decimals=0, dist=("poisson_gamma", 3, 4, 21),
        signal="Psychological up; levothyroxine over-replacement up",
        missing="With PHQ-9, plus ~5% extra gaps", source="Spitzer 2006 Arch Intern Med"),
    Var("qol_instrument", "2", "qol", "Quality-of-life instrument on file", kind="categorical",
        normal="'' / FACT-M / FACT-ICM", observed_only=True,
        missing="~30% have one; 80% FACT-M, 20% FACT-ICM", source="Proposal Tier 2"),
    Var("fact_g", "2", "qol", "FACT-G total (core of FACT-M and FACT-ICM)", "points", "integer",
        "0-108; general population ~80-85", 0, 108, decimals=0, dist=("normal", 82, 12),
        signal="Lower with any burden; psychological most",
        missing="Present whenever a QoL instrument is on file", source="Cella 1993 J Clin Oncol"),
    Var("fact_m_total", "2", "qol", "FACT-M total (FACT-G + melanoma subscale)", "points", "integer",
        "0-172; survivors ~125-140", 0, 172, decimals=0, dist=("normal", 130, 18),
        signal="Lower with any burden; psychological most",
        missing="Present only when the instrument is FACT-M", source="Cormier 2008 Cancer"),
    Var("baseline_cog_z", "2", "baseline_cognition", "Pre-treatment cognitive composite", "z-score",
        normal="mean 0, SD 1", hard_min=-5, hard_max=4, decimals=2, dist=("normal", 0, 1),
        signal="Correlates ~0.75 with current cognitive trait (per-patient baseline)",
        missing="~8% present; ~43% with brain metastases", source="Assumption"),

    # ---------------- Tier 3 ----------------
    Var("tier3_refused", "3", "tier3_admin", "Declined the prospective Tier 3 visit", kind="binary",
        observed_only=True, missing="~20%; more often older or more distressed patients",
        source="Proposal: ~300 complete of ~400"),
    Var("tier3_draw_time", "3", "tier3_admin", "Clock time of Tier 3 blood draw", "HH:MM", "time",
        "07:30-10:30 (90%)", dist=("time", 7.5, 11.0), signal="Afternoon draw lowers testosterone",
        missing="Blank if refused", source="Proposal Tier 3"),
    Var("fasting", "3", "tier3_admin", "Fasting at Tier 3 draw", kind="binary", normal="~85% fasting",
        dist=("binary", 0.85), signal="Non-fasting lowers testosterone ~20-25%",
        missing="Blank if refused", source="Caronia 2013 Clin Endocrinol"),
    Var("igf1", "3", "tier3_blood", "Insulin-like growth factor 1", "ng/mL", normal="age-dependent; ~115-300 at 40, ~65-200 at 70",
        hard_min=5, hard_max=1000, lod=15, decimals=0, dist=("lognormal", 150, 0.42),
        signal="GH deficiency down (age SDS ~N(-1.8,1), overlaps normal); hypothyroidism down",
        missing="Refusal or ~1% assay failure", source="Bidlingmaier 2014 J Clin Endocrinol Metab"),
    Var("lh", "3", "tier3_blood", "Luteinizing hormone", "IU/L", normal="M 1.5-9.3; premenopausal F 2-15 (mid-cycle to ~90); postmenopausal F 15-60",
        hard_min=0.1, hard_max=150, lod=0.3, decimals=1, dist=("lognormal", 6, 0.9),
        signal="Pituitary gonadal deficiency down; testosterone replacement suppresses",
        missing="Refusal or ~1% assay failure", source="Mayo Clinic Laboratories"),
    Var("fsh", "3", "tier3_blood", "Follicle-stimulating hormone", "IU/L", normal="M 1.5-12.4; premenopausal F 3-10; postmenopausal F 25-135",
        hard_min=0.1, hard_max=250, lod=0.3, decimals=1, dist=("lognormal", 10, 1.1),
        signal="Pituitary gonadal deficiency down; postmenopausal pituitary failure <20-25",
        missing="Refusal or ~1% assay failure", source="Mayo Clinic Laboratories"),
    Var("testosterone", "3", "tier3_blood", "Total testosterone (men only; blank for women by design)",
        "ng/dL", normal="264-916", hard_min=2, hard_max=2000, lod=7, decimals=0,
        dist=("lognormal", 480, 0.38),
        signal="Sex-hormone cause down (<250); age, prednisone, non-fasting and afternoon draws lower it",
        missing="Refusal or ~1% assay failure",
        source="Travison 2017 J Clin Endocrinol Metab; Bhasin 2018 J Clin Endocrinol Metab"),
    Var("estradiol", "3", "tier3_blood", "Estradiol (women only; blank for men by design)", "pg/mL",
        normal="premenopausal 30-400 (cycle-dependent); postmenopausal <30",
        hard_min=1, hard_max=1000, lod=5, decimals=0, dist=("lognormal", 25, 1.0),
        signal="Premenopausal sex-hormone cause down", missing="Refusal or ~1% assay failure",
        source="Mayo Clinic Laboratories"),
    Var("il6", "3", "tier3_blood", "Interleukin-6", "pg/mL", normal="<6.4 (assay-dependent; rises with age)",
        hard_min=0.2, hard_max=500, lod=1.0, decimals=1, dist=("lognormal", 2.0, 0.6),
        signal="Inflammation up (6-30); prednisone down", missing="Refusal or ~1% assay failure",
        source="Mayo Clinic Laboratories IL-6 (assay-dependent)"),
    Var("tnf_alpha", "3", "tier3_blood", "Tumor necrosis factor alpha", "pg/mL",
        normal="<=2.8 (median ~1.2; assay-dependent)", hard_min=0.1, hard_max=100, lod=0.5, decimals=1,
        dist=("lognormal", 1.2, 0.4), signal="Inflammation up (2-6)",
        missing="Refusal or ~1% assay failure", source="Mayo Clinic Laboratories TNF-alpha (assay-dependent)"),
    Var("hvlt_total_recall", "3", "tier3_cognition", "HVLT-R total recall (examiner-administered)",
        "words", "integer", "0-36; ~26 +/- 5 at age 60", 0, 36, decimals=0, dist=("normal", 25, 5),
        signal="Follows cognition factor; falls with age", missing="Refusal or ~1% not completed",
        source="Benedict 1998 Clin Neuropsychol; Wefel 2011 Lancet Oncol (ICCTF battery)"),
    Var("tmt_b_seconds", "3", "tier3_cognition", "Trail Making Test B time (higher = worse)", "seconds",
        "integer", "~65-85 +/- 25 at ages 55-64; capped at 300", 20, 300, decimals=0,
        dist=("lognormal", 82, 0.35), signal="Rises as cognition factor falls; rises with age",
        missing="Refusal or ~1% not completed", source="Tombaugh 2004 Arch Clin Neuropsychol"),
    Var("cowa_total", "3", "tier3_cognition", "COWA (FAS) verbal fluency", "words", "integer",
        "~38 +/- 11", 0, 90, decimals=0, dist=("normal", 38, 11), signal="Follows cognition factor",
        missing="Refusal or ~1% not completed", source="Tombaugh 1999 Arch Clin Neuropsychol"),
    Var("pcl5", "3", "tier3_ptsd", "PCL-5 PTSD checklist", "points", "integer",
        "0-80; probable PTSD >=31-33", 0, 80, decimals=0, dist=("poisson_gamma", 10, 2.5, 80),
        signal="Psychological up", missing="Refusal or ~1% not completed",
        source="Blevins 2015 J Trauma Stress; Weathers 2013 (National Center for PTSD)"),

    # ---------------- labels and targets ----------------
    Var("presenting_complaint", "label", "label", "Presenting complaint (drawn the same way for every cause)",
        kind="categorical", normal="fatigue 50% / low_mood 25% / cognitive 25%",
        dist=("choice", ["fatigue", "low_mood", "cognitive"], [0.5, 0.25, 0.25]),
        source="Proposal: all six causes present identically"),
    Var("primary_cause", "label", "label", "Cause with the largest share of symptom burden (6-class label)",
        kind="categorical", normal=" / ".join(CAUSES), source="Generator stage D"),
    Var("secondary_cause", "label", "label", "Second cause if its share is above 0.25 (else blank)",
        kind="categorical", source="Generator stage D"),
] + [
    Var(f"score_{c}", "target", "target", f"Share of symptom burden from {c.replace('_', ' ')} (six scores sum to 1)",
        "fraction", normal="0-1", hard_min=0, hard_max=1, decimals=4, source="Generator stage D")
    for c in CAUSES
]

SPEC = {v.name: v for v in VARIABLES}
OBSERVED_COLUMNS = [v.name for v in VARIABLES]
GROUND_TRUTH_COLUMNS = [v.name for v in VARIABLES if not v.observed_only]

# Hidden physiological states, stored only in the cause-driven ground_truth.csv
HIDDEN = {
    "menstrual_phase": "Premenopausal women: follicular / ovulatory / luteal (explains LH/estradiol spread)",
    "dmg_primary_hypothyroid": "Thyroid gland permanently damaged by thyroiditis",
    "dmg_hypophysitis": "Pituitary inflammation occurred",
    "dmg_central_hypothyroid": "Persistent TSH deficiency after hypophysitis",
    "dmg_acth_deficiency": "Secondary adrenal insufficiency (every hypophysitis; permanent)",
    "dmg_primary_adrenal": "Primary adrenal insufficiency (adrenalitis)",
    "dmg_gh_deficiency": "Persistent GH deficiency after hypophysitis",
    "dmg_gonadal_pituitary": "Persistent gonadotropin deficiency after hypophysitis",
    "dmg_functional_hypogonadism": "Functional low sex hormones (age, steroids, stress)",
    "chronic_inflammatory_irae": "Ongoing inflammatory irAE (arthritis, colitis, etc.)",
    "thyroid_replacement_status": "none / untreated / adequate / under / over",
    "adrenal_replacement_status": "none / untreated / adequate / under",
    "sex_replacement_status": "none / untreated / replaced",
    "severity_thyroid": "Gland damage 0-1 before replacement",
    "severity_adrenal": "Gland damage 0-1 before replacement",
    "severity_gh": "Gland damage 0-1 (never replaced)",
    "severity_sex": "Gland damage 0-1 before replacement",
    "deficit_thyroid": "Current deficit 0-1 after replacement (over-replacement scored as 0.2-0.45)",
    "deficit_adrenal": "Current deficit 0-1 after replacement",
    "deficit_gh": "Current deficit 0-1",
    "deficit_sex": "Current deficit 0-1 after replacement (x0.2 for postmenopausal pituitary loss)",
    "inflammation_factor": "Latent inflammation (SD units; chronic irAE adds 1.5-3)",
    "distress_factor": "Latent psychological distress (SD units)",
    "cognition_factor": "Latent current cognition (SD units; higher is better)",
    **{f"burden_{c}": f"Raw symptom burden from {c.replace('_', ' ')} (score = burden / total)" for c in CAUSES},
    "true_cortisol_0800_endogenous": "Gland cortisol output at 08:00 with no steroid, hydrocortisone, draw-time or assay effect (ug/dL)",
    "hc_hours_since_dose": "Hours since last hydrocortisone dose at the cortisol draw (blank if not on hydrocortisone)",
    "true_igf1_sds": "Age-adjusted IGF-1 standard deviation score",
    "true_testosterone_0800_fasting": "Gland testosterone at 08:00 fasting, before replacement (men, ng/dL)",
}


def sample_marginal(rng, var, n):
    """Draw n values of one column from its population marginal (independent mode)."""
    kind, *p = var.dist
    if kind == "normal":
        x = rng.normal(p[0], p[1], n)
    elif kind == "lognormal":
        x = np.exp(rng.normal(np.log(p[0]), p[1], n))
    elif kind == "gamma":
        x = p[2] + rng.gamma(p[0], p[1], n)
    elif kind == "poisson_gamma":
        x = np.minimum(rng.poisson(p[0] * rng.gamma(p[1], 1 / p[1], n)), p[2]).astype(float)
    elif kind == "binary":
        return (rng.random(n) < p[0]).astype(int)
    elif kind == "choice":
        return rng.choice(np.array(p[0], dtype=object), n, p=p[1])
    elif kind == "time":
        return hours_to_hhmm(rng.uniform(p[0], p[1], n))
    else:
        raise ValueError(f"unknown dist {kind} for {var.name}")
    return x


def hours_to_hhmm(hours):
    minutes = (np.round(np.asarray(hours) * 12) * 5).astype(int)
    return np.array([f"{m // 60:02d}:{m % 60:02d}" for m in minutes], dtype=object)


def hhmm_to_hours(series):
    parts = series.astype("string").str.split(":", expand=True)
    return pd.to_numeric(parts[0], errors="coerce") + pd.to_numeric(parts[1], errors="coerce") / 60


def data_dictionary():
    cols = [f.name for f in fields(Var)]
    rows = [{c: getattr(v, c) for c in cols} for v in VARIABLES]
    for name, desc in HIDDEN.items():
        rows.append({"name": name, "tier": "hidden", "group": "hidden", "description": desc,
                     "missing": "ground_truth.csv only (cause-driven mode)", "observed_only": False})
    df = pd.DataFrame(rows, columns=cols)
    df["dist"] = df["dist"].map(lambda d: "" if not d else repr(d))
    df["in_files"] = np.where(df.tier == "hidden", "ground_truth (cause-driven)",
                              np.where(df.observed_only.astype(bool), "observed", "ground_truth, observed"))
    return df.drop(columns="observed_only")

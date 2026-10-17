"""Generate synthetic data for the checkpoint-inhibitor survivor model.

Modes
  cause        Hidden physiology (gland damage, replacement, inflammation, distress)
               drives every lab and score, the 6-class label and the per-cause scores.
  independent  Every column drawn on its own from its normal range. A null baseline:
               any model should score at chance on it.

Usage
  python3 generate_synthetic_data.py                        # both modes, n=250, seed=42
  python3 generate_synthetic_data.py --priors balanced --n 2000 --signal 0.5

Outputs (under --out, default data/)
  cause_driven/ground_truth.csv   every value, no gaps, plus hidden states
  cause_driven/observed.csv       what a real extract would look like (gaps, "<LOD" strings)
  independent/ground_truth.csv, independent/observed.csv
  data_dictionary.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reference_ranges import (CAUSES, GROUND_TRUTH_COLUMNS, HIDDEN, IGF1_LOG_SD, OBSERVED_COLUMNS,
                              SPEC, VARIABLES, data_dictionary, hours_to_hhmm, igf1_median,
                              sample_marginal)

REGIMENS = ["anti-PD-1", "anti-CTLA-4", "combination"]
REGIMEN_P = [0.60, 0.10, 0.30]
STAGE_P = {"anti-PD-1": [0.15, 0.50, 0.35], "anti-CTLA-4": [0.05, 0.50, 0.45],
           "combination": [0.02, 0.33, 0.65]}

# Gland damage in the whole survivor pool, by regimen
# (Barroso-Sousa 2018 JAMA Oncol; Faje 2019 Eur J Endocrinol; Patrinely 2021 JAMA Oncol)
P_PRIMARY_HYPOTHYROID = {"anti-PD-1": 0.075, "anti-CTLA-4": 0.04, "combination": 0.14}
P_HYPOPHYSITIS = {"anti-PD-1": 0.005, "anti-CTLA-4": 0.09, "combination": 0.07}
P_PRIMARY_ADRENAL = {"anti-PD-1": 0.005, "anti-CTLA-4": 0.007, "combination": 0.03}
# Chance an axis is still deficient 12+ months after hypophysitis = involved x not recovered
# (Nguyen 2021; Albarel 2015; van der Leij 2024). PD-1 hypophysitis is mostly isolated ACTH loss.
P_AXIS_CTLA4 = {"tsh": 0.50 * 0.5, "gonadal": 0.65 * 0.4, "gh": 0.35 * 0.8}
P_AXIS_PD1 = {"tsh": 0.11 * 0.5, "gonadal": 0.10 * 0.4, "gh": 0.0}
# Thyroiditis that resolved to normal function (flag set, no damage)
P_THYROIDITIS_RESOLVED = {"anti-PD-1": 0.04, "anti-CTLA-4": 0.02, "combination": 0.08}
# Non-endocrine irAE history: any-grade rate (PD-1, CTLA-4, combination), CTCAE grade mix [1, 2, 3, 4]
P_IRAE = {
    "colitis": ((0.08, 0.20, 0.25), (0.35, 0.35, 0.25, 0.05)),
    "hepatitis": ((0.05, 0.08, 0.17), (0.40, 0.30, 0.25, 0.05)),
    "pneumonitis": ((0.04, 0.01, 0.07), (0.40, 0.40, 0.17, 0.03)),
    "arthritis": ((0.05, 0.03, 0.07), (0.45, 0.45, 0.10, 0.00)),
    "dermatitis": ((0.30, 0.40, 0.45), (0.60, 0.32, 0.08, 0.00)),
}
ENDOCRINE_GRADE_P = {"thyroiditis": (0.45, 0.45, 0.09, 0.01),
                     "hypophysitis": (0.05, 0.55, 0.35, 0.05),
                     "adrenalitis": (0.00, 0.40, 0.50, 0.10)}

# Class mix of the symptomatic cohort (realistic default; see plan for rationale)
REALISTIC_PRIORS = {"thyroid": 0.20, "adrenal": 0.09, "growth_hormone": 0.02,
                    "sex_hormone": 0.12, "inflammatory": 0.15, "psychological": 0.42}
# How strongly a full deficit drives symptoms, in CAUSES order
BURDEN_WEIGHT = np.array([1.0, 1.1, 0.6, 0.8, 1.0, 1.0])
LT4_TABLETS = np.array([25, 50, 75, 88, 100, 112, 125, 137, 150, 175, 200])
TIER3_ITEMS = ["igf1", "lh", "fsh", "testosterone", "estradiol", "il6", "tnf_alpha",
               "hvlt_total_recall", "tmt_b_seconds", "cowa_total", "pcl5"]


def _by_regimen(ri, table):
    return np.array([table[r] for r in REGIMENS])[ri]


def _choice_rows(rng, levels, probs):
    """One draw per row, where row i uses probability vector probs[i]."""
    cum = np.cumsum(probs, axis=1)
    u = rng.random(len(probs))[:, None]
    return np.asarray(levels, dtype=object)[(u > cum[:, :-1]).sum(axis=1)]


def _severity(rng, n, low):
    return low + (1 - low) * rng.beta(3, 2, n)


def _snap(x, allowed):
    return allowed[np.abs(x[:, None] - allowed[None, :]).argmin(axis=1)]


# ---------------------------------------------------------------- cause-driven: stages A-D
def _latent_pool(rng, n):
    """Patient, gland damage, replacement and symptom burden for a pool of survivors."""
    # A. patient and treatment
    age = np.clip(np.round(rng.normal(61, 13, n)), 25, 90)
    male = rng.random(n) < 0.62
    post = ~male & (rng.random(n) < 1 / (1 + np.exp(-(age - 51) / 2.5)))
    ri = rng.choice(3, n, p=REGIMEN_P)
    ctla = ri != 0
    stage = _choice_rows(rng, ["II", "III", "IV"], np.array([STAGE_P[r] for r in REGIMENS])[ri])
    months = np.clip(np.round(12 + rng.gamma(1.6, 14, n)), 12, 96)
    brain_mets = rng.random(n) < np.where(stage == "IV", 0.15, 0.01)
    radiation = np.where(brain_mets, rng.choice(np.array(["SRS", "WBRT", "none"], dtype=object), n,
                                                p=[0.75, 0.15, 0.10]), "none")

    # B. gland damage
    primary_hypo = rng.random(n) < _by_regimen(ri, P_PRIMARY_HYPOTHYROID)
    hypophysitis = rng.random(n) < _by_regimen(ri, P_HYPOPHYSITIS)

    def axis(k):
        return np.where(ctla, P_AXIS_CTLA4[k], P_AXIS_PD1[k])

    central_hypo = hypophysitis & ~primary_hypo & (rng.random(n) < axis("tsh"))
    primary_adrenal = ~hypophysitis & (rng.random(n) < _by_regimen(ri, P_PRIMARY_ADRENAL))
    gonadal_pit = hypophysitis & (rng.random(n) < axis("gonadal"))
    gh_def = hypophysitis & (rng.random(n) < axis("gh"))

    irae = {
        "thyroiditis": np.where(primary_hypo, rng.random(n) < 0.92,
                                rng.random(n) < _by_regimen(ri, P_THYROIDITIS_RESOLVED)),
        "hypophysitis": np.where(hypophysitis, rng.random(n) < 0.95, rng.random(n) < 0.003),
        "adrenalitis": np.where(primary_adrenal, rng.random(n) < 0.90, rng.random(n) < 0.002),
    }
    for name, (rates, _) in P_IRAE.items():
        irae[name] = rng.random(n) < _by_regimen(ri, dict(zip(REGIMENS, rates)))
    max_grade = np.zeros(n, dtype=int)
    for name, flag in irae.items():
        grade_p = ENDOCRINE_GRADE_P[name] if name in ENDOCRINE_GRADE_P else P_IRAE[name][1]
        max_grade = np.maximum(max_grade, np.where(flag, rng.choice([1, 2, 3, 4], n, p=grade_p), 0))

    chronic = rng.random(n) < (0.05 + 0.45 * irae["arthritis"] + 0.10 * irae["colitis"]
                               + 0.05 * irae["hepatitis"] + 0.05 * irae["pneumonitis"])
    pred = np.where(rng.random(n) < 0.03 + 0.35 * chronic,
                    rng.choice([2.5, 5, 7.5, 10], n, p=[0.3, 0.4, 0.15, 0.15]), 0.0)
    p_functional = np.where(male, np.clip(0.10 + 0.006 * (age - 45), 0.03, 0.35) + 0.15 * (pred >= 5),
                            np.where(post, 0.0, 0.02))
    functional = ~gonadal_pit & (rng.random(n) < p_functional)

    sev_thy = np.where(primary_hypo | central_hypo, _severity(rng, n, 0.3), 0.0)
    sev_adr = np.where(hypophysitis | primary_adrenal, _severity(rng, n, 0.3), 0.0)
    sev_gh = np.where(gh_def, _severity(rng, n, 0.4), 0.0)
    sev_sex = np.select([gonadal_pit, functional], [_severity(rng, n, 0.5), 0.15 + 0.5 * rng.beta(2, 2, n)], 0.0)

    inflam = rng.normal(0, 1, n) + chronic * rng.uniform(1.5, 3.0, n)
    distress = rng.normal(0, 1, n) + 0.25 * (stage == "IV") + 0.4 * brain_mets
    cog_trait = rng.normal(0, 1, n)

    # C. diagnosis and replacement; deficit = damage x (1 - replacement adequacy)
    thy_dmg = primary_hypo | central_hypo
    thy_dx = thy_dmg & (rng.random(n) < np.where(primary_hypo, 0.85, 0.60))
    u = rng.random(n)
    thy_status = np.select([~thy_dmg, ~thy_dx, u < 0.75, u < 0.90],
                           ["none", "untreated", "adequate", "under"], "over")
    # Over-replacement is iatrogenic hyperthyroidism: a thyroid cause with an anxious presentation
    def_thy = np.select([thy_status == "untreated", thy_status == "under", thy_status == "adequate",
                         thy_status == "over"],
                        [sev_thy, sev_thy * rng.uniform(0.35, 0.7, n), sev_thy * 0.05,
                         rng.uniform(0.2, 0.45, n)], 0.0)

    adr_dmg = hypophysitis | primary_adrenal
    # Isolated ACTH loss after PD-1 is insidious and missed more often
    adr_dx = adr_dmg & (rng.random(n) < np.where(primary_adrenal, 0.90, np.where(ctla, 0.92, 0.70)))
    u = rng.random(n)
    adr_status = np.select([~adr_dmg, ~adr_dx, u < 0.75], ["none", "untreated", "adequate"], "under")
    def_adr = np.select([adr_status == "untreated", adr_status == "under", adr_status == "adequate"],
                        [sev_adr, sev_adr * rng.uniform(0.4, 0.8, n), sev_adr * 0.05], 0.0)

    sex_dmg = sev_sex > 0
    sex_rx = sex_dmg & (rng.random(n) < np.where(gonadal_pit, 0.40, 0.20))
    sex_status = np.select([~sex_dmg, sex_rx], ["none", "replaced"], "untreated")
    # Losing gonadotropins after menopause adds little: estrogen is already low
    def_sex = sev_sex * np.where(sex_rx, 0.1, 1.0) * np.where(post & gonadal_pit, 0.2, 1.0)

    lt4_need = np.clip(rng.normal(125, 20, n), 50, 200) * (0.5 + 0.5 * sev_thy)
    lt4 = lt4_need * np.select([thy_status == "under", thy_status == "over"], [0.6, 1.35], 1.0)
    lt4 = np.where(np.isin(thy_status, ["adequate", "under", "over"]), _snap(lt4, LT4_TABLETS), 0)
    hc = np.select([adr_status == "adequate", adr_status == "under"],
                   [rng.choice([15, 20, 25], n, p=[0.35, 0.45, 0.20]),
                    rng.choice([10, 12.5, 15], n, p=[0.45, 0.30, 0.25])], 0.0)

    # D. burdens, symptomatic selection, scores and labels
    b_inf = np.clip((inflam - 1.0) / 2.5, 0, 1)
    b_psy = np.clip((distress - 0.6) / 2.0, 0, 1)
    burden = np.column_stack([def_thy, def_adr, sev_gh, def_sex, b_inf, b_psy]) * BURDEN_WEIGHT
    total = burden.sum(axis=1)
    symptomatic = (total >= 0.05) & (rng.random(n) < 1 - np.exp(-3 * total))
    scores = burden / np.where(total > 0, total, 1)[:, None]
    order = np.argsort(-scores, axis=1)
    second_share = scores[np.arange(n), order[:, 1]]

    L = pd.DataFrame({
        "age": age, "male": male, "post": post, "regimen": np.array(REGIMENS, dtype=object)[ri],
        "stage": stage, "months": months, "brain_mets": brain_mets, "radiation": radiation,
        **{f"irae_{k}": v for k, v in irae.items()}, "irae_max_grade": max_grade,
        "pred": pred, "hc": hc, "lt4": lt4, "sex_rx": sex_rx,
        "dmg_primary_hypothyroid": primary_hypo, "dmg_hypophysitis": hypophysitis,
        "dmg_central_hypothyroid": central_hypo, "dmg_acth_deficiency": hypophysitis,
        "dmg_primary_adrenal": primary_adrenal, "dmg_gh_deficiency": gh_def,
        "dmg_gonadal_pituitary": gonadal_pit, "dmg_functional_hypogonadism": functional,
        "chronic_inflammatory_irae": chronic,
        "thyroid_replacement_status": thy_status, "adrenal_replacement_status": adr_status,
        "sex_replacement_status": sex_status,
        "severity_thyroid": sev_thy, "severity_adrenal": sev_adr, "severity_gh": sev_gh,
        "severity_sex": sev_sex, "deficit_thyroid": def_thy, "deficit_adrenal": def_adr,
        "deficit_gh": sev_gh, "deficit_sex": def_sex,
        "inflammation_factor": inflam, "distress_factor": distress, "cog_trait": cog_trait,
        **{f"burden_{c}": burden[:, k] for k, c in enumerate(CAUSES)},
        **{f"score_{c}": scores[:, k] for k, c in enumerate(CAUSES)},
        "primary_idx": order[:, 0],
        "secondary_idx": np.where(second_share > 0.25, order[:, 1], -1),
        "symptomatic": symptomatic,
    })
    return L


def _target_counts(n, priors):
    p = np.array([priors[c] for c in CAUSES], dtype=float)
    raw = p / p.sum() * n
    counts = np.floor(raw).astype(int)
    counts[np.argsort(counts - raw)[: n - counts.sum()]] += 1
    return counts


def _select_cohort(rng, n, priors, batch=100_000, max_rows=5_000_000):
    """Draw symptomatic survivors in batches, then subsample to the target class mix."""
    target = _target_counts(n, priors)
    parts, have, drawn = [], np.zeros(len(CAUSES), dtype=int), 0
    while (have < target).any():
        if drawn >= max_rows:
            short = [c for c, h, t in zip(CAUSES, have, target) if h < t]
            raise RuntimeError(f"Pool of {drawn:,} survivors has too few cases of {short}; lower --n")
        pool = _latent_pool(rng, batch)
        pool = pool[pool.symptomatic]
        parts.append(pool)
        have += np.bincount(pool.primary_idx, minlength=len(CAUSES))
        drawn += batch
    pool = pd.concat(parts, ignore_index=True)
    labels = pool.primary_idx.to_numpy()
    picks = [rng.choice(np.flatnonzero(labels == k), target[k], replace=False) for k in range(len(CAUSES))]
    return pool.iloc[rng.permutation(np.concatenate(picks))].reset_index(drop=True)


# ---------------------------------------------------------------- cause-driven: stage E
def _measure(rng, L, s):
    """Turn hidden states into lab values and scores. `s` scales every cause-related shift."""
    n = len(L)
    age, male, post = L.age.to_numpy(), L.male.to_numpy(), L.post.to_numpy()
    pre = ~male & ~post
    pred = L.pred.to_numpy()
    pe = np.clip(pred / 5, 0, 2)  # prednisone effect: 0 at none, 1 at 5 mg, 2 at 10 mg
    F, D = L.inflammation_factor.to_numpy(), L.distress_factor.to_numpy()
    thy, adr = L.thyroid_replacement_status.to_numpy(), L.adrenal_replacement_status.to_numpy()
    d_thy, d_adr = L.deficit_thyroid.to_numpy(), L.deficit_adrenal.to_numpy()
    d_gh, d_sex = L.deficit_gh.to_numpy(), L.deficit_sex.to_numpy()
    d_hypo = np.where(thy == "over", 0.0, d_thy)  # thyroid hormone shortfall (not excess)
    sev_adr, sev_sex = L.severity_adrenal.to_numpy(), L.severity_sex.to_numpy()
    b_inf = L.burden_inflammatory.to_numpy()
    primary_hypo, central = L.dmg_primary_hypothyroid.to_numpy(), L.dmg_central_hypothyroid.to_numpy()
    pai, acth_def = L.dmg_primary_adrenal.to_numpy(), L.dmg_acth_deficiency.to_numpy()
    pit, func = L.dmg_gonadal_pituitary.to_numpy(), L.dmg_functional_hypogonadism.to_numpy()
    sex_rx = L.sex_rx.to_numpy()
    on_lt4 = np.isin(thy, ["adequate", "under", "over"])
    on_hc = np.isin(adr, ["adequate", "under"])

    def lognorm(median, sd):
        return median * np.exp(rng.normal(0, sd, n))

    g = {}
    # ---- thyroid
    tsh = lognorm(1.6, 0.45)
    mild = rng.random(n) < np.clip(0.03 + 0.0015 * (age - 40), 0.02, 0.12)  # subclinical, age-related
    tsh = np.where(mild, np.exp(rng.uniform(np.log(4.5), np.log(10), n)), tsh)
    g["tsh"] = np.select(
        [central & on_lt4, central, thy == "over", thy == "adequate", primary_hypo],
        [lognorm(0.05, 0.6),                      # central on levothyroxine: suppressed whatever the dose
         lognorm(1.6 * 0.6 ** s, 0.5),            # central untreated: low or inappropriately normal
         lognorm(1.6 * 0.05 ** s, 0.7),           # over-replaced: suppressed
         lognorm(1.5, 0.9),                       # adequately replaced: ~20% still out of range
         tsh * np.exp(s * 3.2 * d_thy)],          # primary, untreated or under-replaced
        tsh)
    g["free_t4"] = (rng.normal(1.2, 0.17, n) - s * 0.7 * d_hypo
                    + np.where(thy == "adequate", 0.12, 0) + np.where(thy == "over", s * 0.45, 0))

    # ---- blood count and CRP (inflammation factor F, prednisone, adrenal deficit)
    anc = 3.8 * np.exp(s * 0.18 * F + 0.15 * pe + rng.normal(0, 0.28, n))
    alc = 1.8 * np.exp(-s * 0.12 * F - 0.15 * pe + s * 0.12 * d_adr + rng.normal(0, 0.30, n))
    eos = 0.14 * np.exp(s * 0.9 * d_adr - 0.45 * pe + rng.normal(0, 0.55, n))
    mono = 0.5 * np.exp(rng.normal(0, 0.25, n))
    g["wbc"], g["anc"], g["alc"], g["eosinophils"] = anc + alc + eos + mono + 0.03, anc, alc, eos
    g["hemoglobin"] = (np.where(male, 14.9 - 0.012 * (age - 60), 13.4) - s * 0.35 * np.maximum(F, 0)
                       - s * 0.6 * d_hypo - s * 1.0 * d_sex * male + rng.normal(0, 1.0, n))
    g["platelets"] = rng.normal(250 + s * 22 * F, 50)
    g["crp"] = 1.6 * np.exp(0.012 * (age - 60) + s * 0.8 * F - 0.25 * pe + rng.normal(0, 0.8, n))

    # ---- cortisol and ACTH
    draw_h = np.where(rng.random(n) < 0.55, rng.uniform(7, 9, n), rng.uniform(9, 14, n))
    diurnal = np.where(draw_h < 8, 1 + 0.05 * (8 - draw_h), np.exp(-0.1 * (draw_h - 8)))
    endo = lognorm(10.5, 0.30) * np.exp(-s * 2.0 * sev_adr)  # gland output at 08:00
    took_dose = rng.random(n) < 0.7  # otherwise the morning dose was held for the test
    hours_since = np.where(took_dose, rng.uniform(0.5, 4, n), rng.uniform(14, 20, n))
    kinetics = np.where(hours_since < 1, hours_since, np.exp(-0.4 * (hours_since - 1)))
    exogenous = np.where(on_hc, L.hc.to_numpy() * (2 / 3) * 2.2 * kinetics, 0.0)
    # prednisone suppresses the axis and prednisolone cross-reacts in the immunoassay
    g["cortisol_am"] = endo * np.exp(-1.0 * pe) * diurnal + exogenous + 0.25 * pred
    acth_shift = np.select([pai & on_hc, pai, acth_def],
                           [s * 1.8 * sev_adr, s * 2.3 * sev_adr, -s * 1.2 * sev_adr], 0.0)
    g["acth"] = lognorm(22, 0.42) * diurnal * np.exp(acth_shift - 0.6 * pe)
    g["cortisol_draw_time"] = hours_to_hhmm(draw_h)

    # ---- mood, QoL (shared distress factor; endocrine deficits raise somatic items)
    somatic = s * (1.2 * d_thy + 1.2 * d_adr + 0.6 * d_gh + 0.6 * d_sex + 0.5 * b_inf)
    anxiety = s * 0.8 * (thy == "over")
    shared = rng.normal(0, 0.4, n)

    def neg_binomial(mean, k, cap):
        return np.minimum(rng.poisson(mean * rng.gamma(k, 1 / k, n)), cap)

    g["phq9"] = neg_binomial(4 * np.exp(s * 0.75 * D + 0.6 * somatic + shared), 6, 27)
    g["gad7"] = neg_binomial(3 * np.exp(s * 0.65 * D + 0.25 * somatic + anxiety + shared), 6, 21)
    g["pcl5"] = neg_binomial(10 * np.exp(s * 0.65 * D + 0.15 * somatic + 0.8 * shared), 4, 80)
    physical = np.minimum(s * (d_thy + d_adr + 0.6 * d_gh + 0.6 * d_sex + b_inf), 1.5)
    g["fact_g"] = np.clip(np.round(87 - s * 5.5 * D - 12 * physical + rng.normal(0, 7, n)), 0, 108)
    melanoma_subscale = np.clip(np.round(50 - s * 2.5 * D - 5 * physical + rng.normal(0, 4.5, n)), 0, 64)
    g["fact_m_total"] = g["fact_g"] + melanoma_subscale

    # ---- cognition (trait, cause effects, brain radiation as a competing cause)
    cog_trait = L.cog_trait.to_numpy()
    radiation = L.radiation.to_numpy()
    cognition = (cog_trait
                 - s * (0.25 * np.maximum(D, 0) + 0.6 * d_thy + 0.5 * d_adr + 0.3 * d_gh + 0.3 * b_inf)
                 - np.select([radiation == "SRS", radiation == "WBRT"], [0.4, 1.0], 0.0))
    g["baseline_cog_z"] = 0.75 * cog_trait + rng.normal(0, 0.66, n)
    g["hvlt_total_recall"] = 26 - 0.12 * (age - 60) + 4.0 * cognition + rng.normal(0, 2.5, n)
    g["tmt_b_seconds"] = 80 * np.exp(0.012 * (age - 60) - 0.25 * cognition + rng.normal(0, 0.18, n))
    g["cowa_total"] = 39 - 0.08 * (age - 60) + 7 * cognition + rng.normal(0, 6, n)

    # ---- Tier 3 blood
    t3_h = np.where(rng.random(n) < 0.9, rng.uniform(7.5, 10.5, n), rng.uniform(10.5, 13, n))
    fasting = rng.random(n) < 0.85
    g["tier3_draw_time"], g["fasting"] = hours_to_hhmm(t3_h), fasting.astype(int)

    igf1_sds = rng.normal(0, 1, n) - s * 2.2 * d_gh - s * 0.5 * d_hypo - 0.4 * (~male & sex_rx)
    g["igf1"] = igf1_median(age) * np.exp(IGF1_LOG_SD * igf1_sds)

    phase = np.where(pre, rng.choice(np.array(["follicular", "ovulatory", "luteal"], dtype=object), n,
                                     p=[0.5, 0.08, 0.42]), "")
    ovul, luteal = phase == "ovulatory", phase == "luteal"
    gonad_shift = (np.where(pit, -s * 1.5 * sev_sex, 0) + np.where(func & male, -s * 0.3 * sev_sex, 0)
                   - 0.15 * pe + np.where(sex_rx, np.where(male, np.log(0.25), np.log(0.55)), 0))
    lh_median = np.select([male, post, ovul, luteal], [4.5 * np.exp(0.008 * (age - 40)), 30, 35, 4], 5.0)
    fsh_median = np.select([male, post, ovul, luteal], [4.5 * np.exp(0.012 * (age - 40)), 65, 12, 3.5], 6.0)
    g["lh"] = lh_median * np.exp(rng.normal(0, 1, n) * np.select([male, post, ovul], [0.40, 0.40, 0.50], 0.45)
                                 + gonad_shift)
    g["fsh"] = fsh_median * np.exp(rng.normal(0, 1, n) * np.select([male, post], [0.50, 0.45], 0.40)
                                   + gonad_shift)

    t_gland = 560 * np.exp(-0.006 * (age - 40) - s * 1.3 * sev_sex - 0.15 * pe + rng.normal(0, 0.36, n))
    t_drawn = t_gland * (1 - 0.03 * np.maximum(t3_h - 9, 0)) * np.where(fasting, 1.0, 0.8)
    g["testosterone"] = np.where(male, np.where(sex_rx, lognorm(520, 0.45), t_drawn), np.nan)

    e2_median = np.select([post, ovul, luteal], [12, 250, 130], 60)
    e2_shift = (np.where(pit, -s * np.where(post, 0.3, 1.6) * sev_sex, 0)
                + np.where(func & ~male, -s * 1.0 * sev_sex, 0))
    e2 = e2_median * np.exp(rng.normal(0, 1, n) * np.select([ovul, luteal], [0.40, 0.45], 0.50) + e2_shift)
    g["estradiol"] = np.where(male, np.nan, np.where(sex_rx, lognorm(60, 0.5), e2))

    g["il6"] = 2.0 * np.exp(0.01 * (age - 60) + s * 0.5 * F - 0.25 * pe + rng.normal(0, 0.45, n))
    g["tnf_alpha"] = 1.2 * np.exp(0.004 * (age - 60) + s * 0.33 * F - 0.1 * pe + rng.normal(0, 0.33, n))

    # ---- columns copied from the hidden states
    g.update({
        "age": age, "sex": np.where(male, "M", "F"),
        "menopausal_status": np.where(male, None, np.where(post, "post", "pre")),
        "regimen": L.regimen, "stage_at_diagnosis": L.stage, "months_since_therapy_end": L.months,
        **{c: L[c].astype(int) for c in L.columns if c.startswith("irae_")},
        "brain_mets": L.brain_mets.astype(int), "brain_radiation": L.radiation,
        "prednisone_equiv_mg_day": pred, "hydrocortisone_mg_day": L.hc, "levothyroxine_mcg_day": L.lt4,
        "sex_hormone_replacement": sex_rx.astype(int),
        "presenting_complaint": rng.choice(np.array(["fatigue", "low_mood", "cognitive"], dtype=object), n,
                                           p=[0.5, 0.25, 0.25]),
        "primary_cause": np.array(CAUSES, dtype=object)[L.primary_idx],
        "secondary_cause": np.where(L.secondary_idx >= 0, np.array(CAUSES, dtype=object)[L.secondary_idx], None),
        **{f"score_{c}": L[f"score_{c}"] for c in CAUSES},
    })
    gt = pd.DataFrame({c: np.asarray(g[c]) for c in GROUND_TRUTH_COLUMNS if c != "patient_id"})
    gt.insert(0, "patient_id", [f"SYN-{i:05d}" for i in range(1, n + 1)])

    hidden = {c: L[c] for c in HIDDEN if c in L}
    hidden.update({
        "menstrual_phase": phase,
        "cognition_factor": cognition,
        "true_cortisol_0800_endogenous": endo,
        "hc_hours_since_dose": np.where(on_hc, hours_since, np.nan),
        "true_igf1_sds": igf1_sds,
        "true_testosterone_0800_fasting": np.where(male, t_gland, np.nan),
    })
    hidden = pd.DataFrame({c: np.asarray(hidden[c]) for c in HIDDEN})
    for c in hidden.select_dtypes("float").columns:
        hidden[c] = hidden[c].round(4)
    for c in hidden.select_dtypes("bool").columns:
        hidden[c] = hidden[c].astype(int)
    return pd.concat([_finalize(gt), hidden], axis=1)


# ---------------------------------------------------------------- shared helpers
def _finalize(df):
    """Clip to hard limits, round, and give integer columns an integer type."""
    df = df.copy()
    for c in df.columns:
        v = SPEC.get(c)
        if v is None or v.kind not in ("continuous", "integer", "binary"):
            continue
        x = pd.to_numeric(df[c], errors="coerce").astype(float)
        x = x.round(v.decimals if v.kind == "continuous" else 0)
        if v.hard_min is not None or v.hard_max is not None:
            x = x.clip(v.hard_min, v.hard_max)
        df[c] = x.astype("Int64") if (v.kind != "continuous" or v.decimals == 0) else x
    return df


def _observe(rng, gt, L=None):
    """Apply the proposal's missingness rules. MNAR when hidden states L are given, else MCAR."""
    n = len(gt)
    obs = gt[GROUND_TRUTH_COLUMNS].copy()
    cause = L is not None

    def blank(cols, mask):
        obs.loc[np.asarray(mask, dtype=bool), cols] = np.nan

    # Tier 1: in the record, small random gaps
    blank(["tsh", "free_t4"], rng.random(n) < 0.03)
    blank(["wbc", "anc", "alc", "eosinophils", "hemoglobin", "platelets"], rng.random(n) < 0.04)
    blank(["crp"], rng.random(n) < 0.10)

    # Tier 2 cortisol: ordered when adrenal failure is suspected; no draw time counts as missing
    if cause:
        suspicion = ((gt.irae_hypophysitis == 1) | (gt.prednisone_equiv_mg_day > 0)
                     | (gt.hydrocortisone_mg_day > 0)).to_numpy()
        p_order = np.clip(0.15 + 0.70 * L.deficit_adrenal.to_numpy() + 0.30 * suspicion, 0, 0.92)
    else:
        p_order = np.full(n, 0.25)
    ordered = rng.random(n) < p_order
    no_time = ordered & (rng.random(n) < 0.20)
    obs["cortisol_ordered"] = ordered.astype(int)
    obs["cortisol_missing_reason"] = np.where(~ordered, "not_ordered", np.where(no_time, "no_draw_time", None))
    blank(["cortisol_am", "acth", "cortisol_draw_time"], ~ordered | no_time)
    blank(["acth"], rng.random(n) < 0.15)

    # Tier 2 scales: whatever exists
    p_mood = 0.40 + 0.30 * (L.distress_factor.to_numpy() > 0.6) if cause else 0.5
    blank(["phq9", "gad7"], rng.random(n) >= p_mood)
    blank(["gad7"], rng.random(n) < 0.05)
    has_qol = rng.random(n) < 0.30
    icm = has_qol & (rng.random(n) < 0.20)  # FACT-ICM carries FACT-G but not the melanoma subscale
    obs["qol_instrument"] = np.where(~has_qol, None, np.where(icm, "FACT-ICM", "FACT-M"))
    blank(["fact_g", "fact_m_total"], ~has_qol)
    blank(["fact_m_total"], icm)
    p_baseline = 0.08 + 0.35 * gt.brain_mets.to_numpy() if cause else 0.10
    blank(["baseline_cog_z"], rng.random(n) >= p_baseline)

    # Tier 3: refusal (older or more distressed patients refuse more), then ~1% item failures
    if cause:
        logit = np.log(0.15 / 0.85) + 0.03 * (gt.age.to_numpy() - 61) + 0.3 * L.distress_factor.to_numpy()
        p_refuse = 1 / (1 + np.exp(-logit))
    else:
        p_refuse = np.full(n, 0.20)
    refused = rng.random(n) < p_refuse
    obs["tier3_refused"] = refused.astype(int)
    blank(TIER3_ITEMS + ["tier3_draw_time", "fasting"], refused)
    for col in TIER3_ITEMS:
        blank([col], rng.random(n) < 0.01)

    # Results below the detection limit are reported as text, as real extracts do
    for v in VARIABLES:
        if v.lod is None:
            continue
        x = pd.to_numeric(obs[v.name], errors="coerce")
        below = (x < v.lod).fillna(False).to_numpy(dtype=bool)
        col = obs[v.name].astype(object).where(obs[v.name].notna(), None)
        col[below] = f"<{v.lod:g}"
        obs[v.name] = col
    return obs[OBSERVED_COLUMNS]


# ---------------------------------------------------------------- public entry points
def generate_cause(n=250, seed=42, signal=1.0, priors="realistic"):
    rng = np.random.default_rng(seed)
    prior_map = REALISTIC_PRIORS if priors == "realistic" else {c: 1 / len(CAUSES) for c in CAUSES}
    L = _select_cohort(rng, n, prior_map)
    gt = _measure(rng, L, signal)
    return gt, _observe(rng, gt, L)


def generate_independent(n=250, seed=42, priors="realistic"):
    rng = np.random.default_rng([seed, 1])
    prior_map = REALISTIC_PRIORS if priors == "realistic" else {c: 1 / len(CAUSES) for c in CAUSES}
    g = {v.name: sample_marginal(rng, v, n) for v in VARIABLES if v.dist and not v.observed_only}
    male = g["sex"] == "M"
    g["menopausal_status"] = np.where(male, None, g["menopausal_status"])
    g["testosterone"] = np.where(male, g["testosterone"], np.nan)
    g["estradiol"] = np.where(male, np.nan, g["estradiol"])
    # Random label and random scores, with the label's score forced to be the largest
    p = np.array([prior_map[c] for c in CAUSES])
    label = rng.choice(len(CAUSES), n, p=p / p.sum())
    scores = rng.dirichlet(np.ones(len(CAUSES)), n)
    rows = np.arange(n)
    top = scores.argmax(axis=1)
    scores[rows, top], scores[rows, label] = scores[rows, label], scores[rows, top]
    second = np.argsort(-scores, axis=1)[:, 1]
    g["primary_cause"] = np.array(CAUSES, dtype=object)[label]
    g["secondary_cause"] = np.where(scores[rows, second] > 0.25, np.array(CAUSES, dtype=object)[second], None)
    g.update({f"score_{c}": scores[:, k] for k, c in enumerate(CAUSES)})
    gt = pd.DataFrame({c: g[c] for c in GROUND_TRUTH_COLUMNS if c != "patient_id"})
    gt.insert(0, "patient_id", [f"SYN-{i:05d}" for i in range(1, n + 1)])
    gt = _finalize(gt)
    return gt, _observe(rng, gt)


def generate(mode, n=250, seed=42, signal=1.0, priors="realistic"):
    """Return (ground_truth, observed) DataFrames for one mode: 'cause' or 'independent'."""
    if mode == "cause":
        return generate_cause(n, seed, signal, priors)
    return generate_independent(n, seed, priors)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["cause", "independent", "both"], default="both")
    ap.add_argument("--n", type=int, default=250, help="patients per dataset (default 250)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--signal", type=float, default=1.0,
                    help="scale of cause-related shifts: 1 realistic, <1 harder, >1 easier")
    ap.add_argument("--priors", choices=["realistic", "balanced"], default="realistic")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    out = Path(args.out)
    modes = ["cause", "independent"] if args.mode == "both" else [args.mode]
    for mode in modes:
        gt, obs = generate(mode, args.n, args.seed, args.signal, args.priors)
        folder = out / ("cause_driven" if mode == "cause" else "independent")
        folder.mkdir(parents=True, exist_ok=True)
        gt.to_csv(folder / "ground_truth.csv", index=False)
        obs.to_csv(folder / "observed.csv", index=False)
        settings = {"mode": mode, "n": args.n, "seed": args.seed, "priors": args.priors,
                    "signal": args.signal if mode == "cause" else None}
        (folder / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
        mix = gt.primary_cause.value_counts().reindex(CAUSES).fillna(0).astype(int)
        print(f"{folder}: {len(gt)} patients | " + ", ".join(f"{c} {k}" for c, k in mix.items()))
    out.mkdir(parents=True, exist_ok=True)
    data_dictionary().to_csv(out / "data_dictionary.csv", index=False)
    print(f"{out / 'data_dictionary.csv'}: {len(data_dictionary())} variables")


if __name__ == "__main__":
    main()

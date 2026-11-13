"""Generate completely random synthetic data for the checkpoint-inhibitor survivor model.

Every column is drawn on its own from its typical adult range (see reference_ranges.py).
Nothing depends on anything else: labs do not depend on the cause, and the cause label
and per-cause scores are random. A model trained on this data should score at chance.
The data is for building and testing the pipeline before the MSK data arrives.

The only rules that tie columns together are structural:
  - testosterone is blank for women; estradiol and menopausal status are blank for men
  - the six per-cause scores sum to 1, and the primary cause has the largest score

Usage
  python3 generate_synthetic_data.py                      # n=250, seed=42
  python3 generate_synthetic_data.py --n 400 --seed 7 --priors balanced

Outputs (under --out, default data/)
  ground_truth.csv     every value, no gaps
  observed.csv         the same patients the way a real extract looks: random gaps, "<LOD" strings
  settings.json        the settings used
  data_dictionary.csv  units, ranges, distributions, sources
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reference_ranges import (CAUSES, GROUND_TRUTH_COLUMNS, OBSERVED_COLUMNS, SPEC, VARIABLES,
                              data_dictionary, sample_marginal)

# Class mix the random labels are drawn from. "balanced" gives 1/6 each.
REALISTIC_PRIORS = {"thyroid": 0.20, "adrenal": 0.09, "growth_hormone": 0.02,
                    "sex_hormone": 0.12, "inflammatory": 0.15, "psychological": 0.42}
TIER3_ITEMS = ["igf1", "lh", "fsh", "testosterone", "estradiol", "il6", "tnf_alpha",
               "hvlt_total_recall", "tmt_b_seconds", "cowa_total", "pcl5"]


def _finalize(df):
    """Round, clip to hard limits, and give integer columns an integer type."""
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


def _observe(rng, gt):
    """Punch gaps into a copy of the data. Every gap is random (missing completely at random)."""
    n = len(gt)
    obs = gt[GROUND_TRUTH_COLUMNS].copy()

    def blank(cols, mask):
        obs.loc[np.asarray(mask, dtype=bool), cols] = np.nan

    # Tier 1: in the record, small random gaps
    blank(["tsh", "free_t4"], rng.random(n) < 0.03)
    blank(["wbc", "anc", "alc", "eosinophils", "hemoglobin", "platelets"], rng.random(n) < 0.04)
    blank(["crp"], rng.random(n) < 0.10)

    # Tier 2 cortisol: ordered for ~25%; no draw time counts as missing (proposal data conditions)
    ordered = rng.random(n) < 0.25
    no_time = ordered & (rng.random(n) < 0.20)
    obs["cortisol_ordered"] = ordered.astype(int)
    obs["cortisol_missing_reason"] = np.where(~ordered, "not_ordered", np.where(no_time, "no_draw_time", None))
    blank(["cortisol_am", "acth", "cortisol_draw_time"], ~ordered | no_time)
    blank(["acth"], rng.random(n) < 0.15)

    # Tier 2 scales: whatever exists
    blank(["phq9", "gad7"], rng.random(n) >= 0.50)
    blank(["gad7"], rng.random(n) < 0.05)
    has_qol = rng.random(n) < 0.30
    icm = has_qol & (rng.random(n) < 0.20)  # FACT-ICM carries FACT-G but not the melanoma subscale
    obs["qol_instrument"] = np.where(~has_qol, None, np.where(icm, "FACT-ICM", "FACT-M"))
    blank(["fact_g", "fact_m_total"], ~has_qol)
    blank(["fact_m_total"], icm)
    blank(["baseline_cog_z"], rng.random(n) >= 0.10)

    # Tier 3: ~20% refuse the whole visit, then ~1% failure per item
    refused = rng.random(n) < 0.20
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


def generate(n=250, seed=42, priors="realistic"):
    """Return (ground_truth, observed) DataFrames of n random patients."""
    rng = np.random.default_rng(seed)
    prior_map = REALISTIC_PRIORS if priors == "realistic" else {c: 1 / len(CAUSES) for c in CAUSES}

    # Every measured column on its own, from its typical range
    g = {v.name: sample_marginal(rng, v, n) for v in VARIABLES if v.dist and not v.observed_only}
    male = g["sex"] == "M"
    g["menopausal_status"] = np.where(male, None, g["menopausal_status"])
    g["testosterone"] = np.where(male, g["testosterone"], np.nan)
    g["estradiol"] = np.where(male, np.nan, g["estradiol"])

    # Random label and random scores; the label's score is swapped into the top spot
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=250, help="number of patients (default 250)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--priors", choices=["realistic", "balanced"], default="realistic",
                    help="class mix of the random labels")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gt, obs = generate(args.n, args.seed, args.priors)
    gt.to_csv(out / "ground_truth.csv", index=False)
    obs.to_csv(out / "observed.csv", index=False)
    settings = {"n": args.n, "seed": args.seed, "priors": args.priors}
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    data_dictionary().to_csv(out / "data_dictionary.csv", index=False)
    mix = gt.primary_cause.value_counts().reindex(CAUSES).fillna(0).astype(int)
    print(f"{out}: {len(gt)} random patients | " + ", ".join(f"{c} {k}" for c, k in mix.items()))


if __name__ == "__main__":
    main()

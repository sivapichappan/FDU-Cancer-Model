"""Check the random synthetic dataset and write data/validation_report.txt.

Run after generate_synthetic_data.py. Exits with status 1 if any check fails.
The data is meant to be completely random, so the checks confirm two things:
the values look like real records (ranges, limits, gaps), and there is no pattern
in them (no column predicts the cause, columns are unrelated). The logistic and
ridge fits here are that check, not the modeling step.

Usage
  python3 validate_synthetic_data.py            # reads data/
  python3 validate_synthetic_data.py --data other_folder
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import generate_synthetic_data as gen
from reference_ranges import (CAUSES, GROUND_TRUTH_COLUMNS, OBSERVED_COLUMNS, SPEC, VARIABLES,
                              hhmm_to_hours)

SCORE_COLS = [f"score_{c}" for c in CAUSES]
FEATURES = [v for v in VARIABLES if v.tier not in ("id", "label", "target")]
NUMERIC = [v.name for v in FEATURES if v.kind in ("continuous", "integer")]
TIER3_COMMON = ["igf1", "lh", "fsh", "il6", "tnf_alpha", "hvlt_total_recall", "tmt_b_seconds",
                "cowa_total", "pcl5"]
# Columns that are blank in ground_truth.csv by design
BLANK_BY_DESIGN = ["menopausal_status", "testosterone", "estradiol", "secondary_cause"]
# Feature groups for the co-occurrence audit (one marker column each)
AUDIT_GROUPS = {"thyroid labs": "tsh", "CBC": "wbc", "CRP": "crp", "cortisol": "cortisol_am",
                "PHQ-9/GAD-7": "phq9", "QoL": "fact_g", "baseline cognition": "baseline_cog_z"}
TIER2_GROUPS = ["cortisol", "PHQ-9/GAD-7", "QoL", "baseline cognition"]


class Report:
    def __init__(self):
        self.lines, self.failed = [], 0

    def section(self, title):
        self.lines += ["", title, "-" * len(title)]

    def line(self, text=""):
        self.lines.append(text)

    def table(self, df):
        self.lines += df.to_string().splitlines()

    def check(self, name, ok, detail=""):
        self.lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
        self.failed += not ok


def parse_lod(col):
    """'<0.3' -> 0.15 (half the detection limit); everything else -> float."""
    s = col.astype("string")
    below = s.str.startswith("<", na=False)
    vals = pd.to_numeric(s.str.lstrip("<"), errors="coerce").astype(float)
    return vals.where(~below, vals / 2)


def single_column(df, v):
    """One feature as numbers: category codes, clock hours, or parsed values."""
    col = df[v.name]
    if v.kind == "categorical":
        return pd.Series(pd.Categorical(col).codes, index=df.index, dtype=float).replace(-1, np.nan)
    if v.kind == "time":
        return hhmm_to_hours(col)
    return parse_lod(col)


def feature_matrix(df):
    """Numeric matrix for the no-signal fits: one-hot categories, gaps median-filled plus a missing flag."""
    parts = []
    for v in FEATURES:
        if v.kind == "categorical":
            parts.append(pd.get_dummies(df[v.name].fillna("missing"), prefix=v.name, dtype=float))
        else:
            parts.append(single_column(df, v).rename(v.name).to_frame())
    X = pd.concat(parts, axis=1)
    for c in X.columns[X.isna().any()]:
        X[c + "_missing"] = X[c].isna().astype(float)
        X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0)
    return X


def cv_accuracy(df):
    X, y = feature_matrix(df), df.primary_cause
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tiny classes trigger fold-size warnings
        return cross_val_score(model, X, y, cv=folds).mean()


def cv_r2(df):
    X = feature_matrix(df)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 9)))  # strength tuned inside each fold
    folds = KFold(5, shuffle=True, random_state=0)
    return np.mean([cross_val_score(model, X, df[c].astype(float), cv=folds, scoring="r2").mean()
                    for c in SCORE_COLS])


def leakage_scan(obs, threshold=0.75, min_cases=20):
    """One-feature depth-3 tree per column; per-class one-vs-rest AUC from 5-fold predictions.

    Classes with fewer than `min_cases` patients are reported but not judged: with a handful
    of cases the AUC swings widely by chance.
    """
    y = obs.primary_cause.to_numpy()
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    rows = []
    for v in FEATURES:
        x = single_column(obs, v).fillna(-999).to_numpy().reshape(-1, 1)
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = cross_val_predict(tree, x, y, cv=folds, method="predict_proba")
        for k, c in enumerate(np.unique(y)):
            rows.append((v.name, c, roc_auc_score(y == c, proba[:, k])))
    res = pd.DataFrame(rows, columns=["feature", "class", "auc"]).sort_values("auc", ascending=False)
    res["cases"] = res["class"].map(obs.primary_cause.value_counts())
    judged = res[res.cases >= min_cases]
    return res, judged[judged.auc > threshold]


def check_limits(r, df):
    bad = []
    for v in VARIABLES:
        if v.name not in df or (v.hard_min is None and v.hard_max is None):
            continue
        x = parse_lod(df[v.name]).dropna()
        if (v.hard_min is not None and (x < v.hard_min).any()) or (v.hard_max is not None and (x > v.hard_max).any()):
            bad.append(f"{v.name} {x.min():g}-{x.max():g}")
    r.check("every value within hard physiological limits", not bad, "; ".join(bad))


def cooccurrence_audit(r, obs):
    present = pd.DataFrame({g: obs[c].notna() for g, c in AUDIT_GROUPS.items()})
    n_groups = present.sum(axis=1)
    n_tier2 = present[TIER2_GROUPS].sum(axis=1)
    r.line(f"Patients with 2+ Tier 1/2 feature groups at the visit: {(n_groups >= 2).sum()} of {len(obs)}")
    r.line(f"Patients with 1+ Tier 2 group beside the Tier 1 labs: {(n_tier2 >= 1).sum()}; "
           f"with 2+ Tier 2 groups: {(n_tier2 >= 2).sum()}")
    r.line("Number of Tier 2 groups present per patient:")
    r.table(n_tier2.value_counts().sort_index().rename("patients").to_frame())
    r.line("Pairwise co-occurrence (patients with both groups):")
    r.table(present.astype(int).T.dot(present.astype(int)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    data = Path(ap.parse_args().data)
    gt = pd.read_csv(data / "ground_truth.csv")
    obs = pd.read_csv(data / "observed.csv", dtype=str)
    settings = json.loads((data / "settings.json").read_text())
    n = len(gt)
    r = Report()
    r.line(f"Random synthetic data validation: n={n}, seed={settings['seed']}, priors={settings['priors']}")

    r.section("1. Files and structure")
    r.check("ground_truth.csv and observed.csv have the expected columns",
            list(gt.columns) == GROUND_TRUTH_COLUMNS and list(obs.columns) == OBSERVED_COLUMNS)
    fresh_gt, fresh_obs = gen.generate(settings["n"], settings["seed"], settings["priors"])
    same = (fresh_gt.to_csv(index=False) == (data / "ground_truth.csv").read_text()
            and fresh_obs.to_csv(index=False) == (data / "observed.csv").read_text())
    r.check("files match a fresh run with the saved settings (reproducible)", same)
    gaps = gt.drop(columns=BLANK_BY_DESIGN).isna().sum()
    r.check("ground_truth.csv has no gaps (apart from the by-design blanks)", gaps.sum() == 0,
            ", ".join(f"{c} {k}" for c, k in gaps[gaps > 0].items()))
    male = gt.sex == "M"
    sex_ok = ((gt.testosterone.notna() == male) & (gt.estradiol.notna() == ~male)
              & (gt.menopausal_status.notna() == ~male)).all()
    r.check("testosterone only for men; estradiol and menopausal status only for women", sex_ok)
    check_limits(r, gt)
    S = gt[SCORE_COLS].to_numpy(dtype=float)
    label_idx = gt.primary_cause.map(CAUSES.index).to_numpy()
    r.check("per-cause scores sum to 1 and the primary cause has the top score",
            np.abs(S.sum(axis=1) - 1).max() < 2e-3 and (S[np.arange(n), label_idx] >= S.max(axis=1) - 1e-9).all())
    mix = gt.primary_cause.value_counts().reindex(CAUSES).fillna(0).astype(int)
    r.line("Class mix (random labels):")
    r.table(pd.DataFrame({"patients": mix, "share": (mix / n).round(2)}))

    r.section("2. Value ranges (ground truth; compare with the 'normal' column of the data dictionary)")
    num = gt[NUMERIC].astype(float)
    ranges = num.describe(percentiles=[0.05, 0.5, 0.95]).T[["min", "5%", "50%", "95%", "max"]]
    ranges["normal"] = [SPEC[c].normal[:45] for c in ranges.index]
    r.table(ranges.round(2))

    r.section("3. Missingness (observed.csv; every gap is random)")
    orphan = (obs.cortisol_am.notna() & obs.cortisol_draw_time.isna()).sum()
    r.check("no cortisol without a draw time", orphan == 0, f"{orphan} rows")
    sex_item = np.where(obs.sex == "M", obs.testosterone.notna(), obs.estradiol.notna())
    complete = int((obs[TIER3_COMMON].notna().all(axis=1) & sex_item).sum())
    expect = 0.8 * 0.99 ** 10 * n
    r.check(f"complete Tier 3 rows near {expect:.0f}", abs(complete - expect) <= 0.05 * n + 5, f"{complete} of {n}")
    present = obs.notna().mean().round(2)
    r.line("Share present by column (testosterone, estradiol and menopausal_status are blank for one sex by design):")
    r.table(present[present < 1].rename("share present").to_frame())
    lod_counts = {v.name: int(obs[v.name].astype("string").str.startswith("<", na=False).sum())
                  for v in VARIABLES if v.lod is not None}
    r.line("Below-detection-limit strings: " + ", ".join(f"{k} {c}" for k, c in lod_counts.items() if c))
    r.line("")
    r.line("Co-occurrence audit (mirrors the proposal's first deliverable):")
    cooccurrence_audit(r, obs)

    r.section("4. No pattern in the data")
    corr = num.corr(method="spearman")
    iu = np.triu_indices(len(corr), 1)
    pairs = pd.DataFrame({"a": corr.index[iu[0]], "b": corr.columns[iu[1]], "r": corr.to_numpy()[iu]}).dropna()
    mean_abs = pairs.r.abs().mean()
    r.check("columns are unrelated: average |Spearman r| near the pure-noise level", mean_abs <= 0.08,
            f"{mean_abs:.3f}; noise alone gives ~{np.sqrt(2 / np.pi) / np.sqrt(n - 1):.3f}")
    r.line("Largest chance correlations (for review; values up to ~0.3 happen by chance at this size):")
    r.table(pairs.reindex(pairs.r.abs().sort_values(ascending=False).index).head(5).round(3).reset_index(drop=True))
    res, flagged = leakage_scan(obs)
    small = sorted(res.loc[res.cases < 20, "class"].unique())
    r.check("no single column predicts the cause (one-vs-rest AUC <= 0.75, classes with 20+ cases)",
            flagged.empty, "; ".join(f"{a} -> {b} {c:.2f}" for a, b, c, _ in flagged.itertuples(index=False)))
    if small:
        r.line(f"Too few cases to judge (shown, not tested): {', '.join(small)}")
    r.line("Strongest single-column results (chance level is 0.5):")
    r.table(res.head(5).round(3).reset_index(drop=True))
    maj = obs.primary_cause.value_counts(normalize=True).max()
    acc, r2 = cv_accuracy(obs), cv_r2(obs)
    r.check("all columns together: logistic accuracy no better than always guessing the biggest class",
            acc <= maj + 0.06, f"{acc:.2f} vs {maj:.2f}")
    r.check("all columns together: ridge R^2 on per-cause scores ~0 or below", r2 <= 0.05, f"mean R^2 {r2:.2f}")
    r.line("On pure noise, accuracy below the majority rate and negative R^2 are expected (the fit chases noise).")

    r.section("Result")
    r.line("ALL CHECKS PASSED" if r.failed == 0 else f"{r.failed} CHECK(S) FAILED")
    text = "\n".join(r.lines) + "\n"
    (data / "validation_report.txt").write_text(text)
    print(text)
    sys.exit(1 if r.failed else 0)


if __name__ == "__main__":
    main()

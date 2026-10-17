"""Check the synthetic datasets and write data/validation_report.txt.

Run after generate_synthetic_data.py. Exits with status 1 if any check fails.
The logistic and ridge fits here only confirm the data carries learnable signal
(and the null mode does not); they are not the modeling step.

Usage
  python3 validate_synthetic_data.py            # reads data/
  python3 validate_synthetic_data.py --data other_folder
"""
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
from reference_ranges import CAUSES, VARIABLES, hhmm_to_hours, igf1_sds

SCORE_COLS = [f"score_{c}" for c in CAUSES]
FEATURES = [v for v in VARIABLES if v.tier not in ("id", "label", "target")]
TIER3_COMMON = ["igf1", "lh", "fsh", "il6", "tnf_alpha", "hvlt_total_recall", "tmt_b_seconds",
                "cowa_total", "pcl5"]
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
    """Numeric matrix for the sanity fits: one-hot categories, gaps median-filled plus a missing flag."""
    parts = []
    for v in FEATURES:
        if v.name not in df:
            continue
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
    return np.mean([cross_val_score(model, X, df[c], cv=folds, scoring="r2").mean() for c in SCORE_COLS])


def check_limits(r, df, label):
    bad = []
    for v in VARIABLES:
        if v.name not in df or (v.hard_min is None and v.hard_max is None):
            continue
        x = parse_lod(df[v.name]).dropna()
        if (v.hard_min is not None and (x < v.hard_min).any()) or (v.hard_max is not None and (x > v.hard_max).any()):
            bad.append(f"{v.name} {x.min():g}-{x.max():g}")
    r.check(f"{label}: every value within hard physiological limits", not bad, "; ".join(bad))


def check_scores(r, df, label):
    S = df[SCORE_COLS].to_numpy(dtype=float)
    label_idx = df.primary_cause.map(CAUSES.index).to_numpy()
    sums_ok = np.abs(S.sum(axis=1) - 1).max() < 2e-3
    argmax_ok = (S[np.arange(len(df)), label_idx] >= S.max(axis=1) - 1e-9).all()
    r.check(f"{label}: per-cause scores sum to 1 and the label has the top score", sums_ok and argmax_ok)


def check_direction(r, gt):
    gt = gt.assign(igf1_sds=igf1_sds(gt.igf1.astype(float), gt.age.astype(float)))
    men = gt.sex == "M"
    not_on_hc = gt.hydrocortisone_mg_day == 0
    expectations = [  # (column, class, highest or lowest, rows to use, note)
        ("tsh", "thyroid", "max", None, ""), ("free_t4", "thyroid", "min", None, ""),
        ("cortisol_am", "adrenal", "min", not_on_hc, "patients not on hydrocortisone"),
        ("eosinophils", "adrenal", "max", None, ""),
        ("igf1_sds", "growth_hormone", "min", None, "age-adjusted"),
        ("testosterone", "sex_hormone", "min", men, "men"),
        ("crp", "inflammatory", "max", None, ""), ("il6", "inflammatory", "max", None, ""),
        ("tnf_alpha", "inflammatory", "max", None, ""),
        ("phq9", "psychological", "max", None, ""), ("gad7", "psychological", "max", None, ""),
        ("pcl5", "psychological", "max", None, ""),
    ]
    for col, cause, way, rows, note in expectations:
        sub = gt if rows is None else gt[rows]
        med = sub.groupby("primary_cause")[col].median().dropna()
        target = med.get(cause)
        others = med.drop(cause, errors="ignore")
        ok = target is not None and (target > others.max() if way == "max" else target < others.min())
        best_other = others.idxmax() if way == "max" else others.idxmin()
        detail = f"{cause} {target:g} vs next {best_other} {others[best_other]:g}" + (f"; {note}" if note else "")
        r.check(f"{col} median is {'highest' if way == 'max' else 'lowest'} in {cause}", ok, detail)


def leakage_scan(obs, threshold=0.95, min_cases=20):
    """One-feature depth-3 tree per column; per-class one-vs-rest AUC from 5-fold predictions.

    Classes with fewer than `min_cases` patients are reported but not judged: with a handful
    of positives the AUC swings by 0.2+ between seeds.
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
        classes = np.unique(y)
        for k, c in enumerate(classes):
            rows.append((v.name, c, roc_auc_score(y == c, proba[:, k])))
    res = pd.DataFrame(rows, columns=["feature", "class", "auc"]).sort_values("auc", ascending=False)
    counts = obs.primary_cause.value_counts()
    res["cases"] = res["class"].map(counts)
    judged = res[res.cases >= min_cases]
    return res, judged[judged.auc > threshold]


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
    r = Report()
    load = {}
    for mode, folder in [("cause", "cause_driven"), ("independent", "independent")]:
        d = data / folder
        load[mode] = (pd.read_csv(d / "ground_truth.csv"), pd.read_csv(d / "observed.csv", dtype=str),
                      json.loads((d / "settings.json").read_text()))
    gt_c, obs_c, set_c = load["cause"]
    gt_i, obs_i, set_i = load["independent"]
    n = len(gt_c)
    r.line(f"Synthetic data validation: n={n}, seed={set_c['seed']}, signal={set_c['signal']}, priors={set_c['priors']}")

    r.section("1. Structure and limits")
    r.check("observed.csv has the same columns in both modes", list(obs_c.columns) == list(obs_i.columns))
    for label, gt in [("cause-driven", gt_c), ("independent", gt_i)]:
        check_limits(r, gt, label)
        check_scores(r, gt, label)
    mix = pd.DataFrame({"cause-driven": gt_c.primary_cause.value_counts(),
                        "independent": gt_i.primary_cause.value_counts()}).reindex(CAUSES).fillna(0).astype(int)
    r.line("Class mix:")
    r.table(mix)

    r.section("2. Cause signatures (cause-driven ground truth, medians)")
    check_direction(r, gt_c)
    show = ["tsh", "free_t4", "eosinophils", "crp", "cortisol_am", "acth", "phq9", "gad7", "pcl5",
            "fact_m_total", "igf1", "lh", "fsh", "testosterone", "estradiol", "il6", "tnf_alpha",
            "hvlt_total_recall", "tmt_b_seconds", "cowa_total"]
    summary = gt_c.groupby("primary_cause")[show].median().reindex(CAUSES).T
    summary.columns = [c[:12] for c in summary.columns]
    summary.insert(0, "min", gt_c[show].min())
    summary.insert(1, "max", gt_c[show].max())
    r.table(summary.round(2))
    non = {c: gt_c[gt_c.primary_cause != c] for c in CAUSES}
    men_non_sex = non["sex_hormone"][non["sex_hormone"].sex == "M"]
    endo = gt_c[gt_c.primary_cause.isin(CAUSES[:4])]
    r.line("Overlap outside the matching class: "
           f"TSH 4.5-10 {((non['thyroid'].tsh > 4.5) & (non['thyroid'].tsh <= 10)).mean():.0%}, "
           f"testosterone <300 in men {(men_non_sex.testosterone < 300).mean():.0%}, "
           f"CRP >5 {(non['inflammatory'].crp > 5).mean():.0%}, "
           f"PHQ-9 >=10 in endocrine classes {(endo.phq9 >= 10).mean():.0%}")
    rho = gt_c[["phq9", "gad7", "pcl5"]].corr(method="spearman").to_numpy()[np.triu_indices(3, 1)]
    r.line(f"Distress scales Spearman r: {rho.min():.2f}-{rho.max():.2f}; "
           f"secondary cause present in {gt_c.secondary_cause.notna().mean():.0%}")

    r.section("3. Missingness (observed.csv)")
    ordered = obs_c.cortisol_ordered.astype(int)
    rate = ordered.groupby(obs_c.primary_cause).mean().reindex(CAUSES)
    others = ordered[obs_c.primary_cause != "adrenal"].mean()
    r.check("cortisol ordered far more often in the adrenal class (MNAR)", rate["adrenal"] > 2 * others,
            f"adrenal {rate['adrenal']:.0%} vs others {others:.0%}")
    for label, obs in [("cause-driven", obs_c), ("independent", obs_i)]:
        orphan = (obs.cortisol_am.notna() & obs.cortisol_draw_time.isna()).sum()
        r.check(f"{label}: no cortisol without a draw time", orphan == 0, f"{orphan} rows")
        sex_item = np.where(obs.sex == "M", obs.testosterone.notna(), obs.estradiol.notna())
        complete = int((obs[TIER3_COMMON].notna().all(axis=1) & sex_item).sum())
        expect = 0.8 * 0.99 ** 10 * len(obs)
        r.check(f"{label}: complete Tier 3 rows near {expect:.0f}", abs(complete - expect) <= 0.05 * len(obs) + 5,
                f"{complete} of {len(obs)}")
    present = obs_c.notna().mean().round(2)
    r.line("Share present by column, cause-driven (testosterone, estradiol, menopausal_status are "
           "blank for one sex by design):")
    r.table(present[present < 1].rename("share present").to_frame())
    lod_counts = {v.name: int(obs_c[v.name].astype("string").str.startswith("<", na=False).sum())
                  for v in VARIABLES if v.lod is not None}
    r.line("Below-detection-limit strings: " + ", ".join(f"{k} {c}" for k, c in lod_counts.items() if c))

    r.section("4. Co-occurrence audit (cause-driven observed, mirrors the proposal's first deliverable)")
    cooccurrence_audit(r, obs_c)

    r.section("5. Leakage scan: one feature at a time (cause-driven observed)")
    res, flagged = leakage_scan(obs_c)
    small = sorted(res.loc[res.cases < 20, "class"].unique())
    r.check("no single feature nearly determines a class (one-vs-rest AUC <= 0.95, classes with 20+ cases)",
            flagged.empty, "; ".join(f"{a} -> {b} {c:.2f}" for a, b, c, _ in flagged.itertuples(index=False)))
    if small:
        r.line(f"Too few cases to judge (shown, not tested): {', '.join(small)}")
    r.line("Strongest single-feature signals (for review):")
    r.table(res.head(10).round(3).reset_index(drop=True))

    r.section("6. Learnable signal (5-fold CV on observed.csv; sanity check, not the model)")
    maj_c = obs_c.primary_cause.value_counts(normalize=True).max()
    maj_i = obs_i.primary_cause.value_counts(normalize=True).max()
    acc_c, acc_i = cv_accuracy(obs_c), cv_accuracy(obs_i)
    r2_c, r2_i = cv_r2(obs_c), cv_r2(obs_i)
    r.check("independent mode: logistic accuracy near the majority-class rate", acc_i <= maj_i + 0.06,
            f"{acc_i:.2f} vs majority {maj_i:.2f}")
    r.check("cause-driven mode: logistic accuracy well above majority", acc_c >= maj_c + 0.10,
            f"{acc_c:.2f} vs majority {maj_c:.2f}")
    r.check("independent mode: ridge R^2 on per-cause scores ~0", r2_i <= 0.05, f"mean R^2 {r2_i:.2f}")
    r.check("cause-driven mode: ridge R^2 on per-cause scores clearly positive", r2_c >= 0.15, f"mean R^2 {r2_c:.2f}")
    r.line("On pure noise, accuracy below the majority rate and negative R^2 are expected (the fit chases noise).")
    sweep = {}
    for s in (0.5, 1.0, 2.0):
        _, obs_s = gen.generate("cause", n, set_c["seed"], s, set_c["priors"])
        sweep[s] = cv_accuracy(obs_s)
    acc = list(sweep.values())
    r.check("accuracy rises with --signal", acc[0] < acc[1] < acc[2],
            ", ".join(f"signal {s}: {a:.2f}" for s, a in sweep.items()))

    r.section("Result")
    r.line("ALL CHECKS PASSED" if r.failed == 0 else f"{r.failed} CHECK(S) FAILED")
    text = "\n".join(r.lines) + "\n"
    (data / "validation_report.txt").write_text(text)
    print(text)
    sys.exit(1 if r.failed else 0)


if __name__ == "__main__":
    main()

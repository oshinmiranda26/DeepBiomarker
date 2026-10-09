"""Responsible-AI audit: subgroup performance and calibration, plus an auto-generated model card.

A model can look good overall and still fail a group of patients. This audit reports, for each subgroup
(sex, age group, neighborhood deprivation, housing instability):
  - discrimination: AUROC with a bootstrap 95% interval, and AUPRC
  - calibration: observed/expected events ratio (1.0 = predicted risk matches reality) and calibration slope
    (1.0 = ideal; below 1 means predictions are too extreme)
  - the oracle AUROC (from the true simulated risk), which separates "this group is inherently harder to
    predict" from "the model is failing this group"

Run:  python -m deepbiomarker.audit        (writes results/subgroup_audit.csv, a calibration plot, and MODEL_CARD.md)
"""
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from deepbiomarker.contributions import lr_from_arrays
from deepbiomarker.dataset import FEATURE_SETS, build_sequences, build_vocab, load, split_patients, static_matrix

RESULTS, MODELS = Path("results"), Path("models")
FLAG_AUROC_GAP = 0.05          # flag a subgroup whose AUROC trails the overall AUROC by more than this
FLAG_OE = (0.8, 1.25)          # flag observed/expected ratios outside this range


def subgroups(patients):
    adi = pd.cut(patients.adi_percentile, [-0.1, 33.3, 66.7, 100], labels=["low", "middle", "high"])
    age = pd.cut(patients.age, [0, 44, 64, 200], labels=["18-44", "45-64", "65+"])
    return {"sex": patients.sex, "age group": age.astype(str), "neighborhood deprivation": adi.astype(str),
            "housing instability": patients.housing_instability.map({0: "no", 1: "yes"})}


def calibration(y, p, eps=1e-6):
    """Observed/expected ratio and calibration slope (logistic regression of the outcome on logit(p))."""
    oe = y.mean() / p.mean()
    logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps)).reshape(-1, 1)
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y).coef_[0, 0] if 0 < y.sum() < len(y) else np.nan
    return oe, slope


def recalibrator(y_val, p_val, eps=1e-6):
    """Platt scaling: logistic regression of the outcome on logit(p), fit on validation data only."""
    to_logit = lambda p: np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=1000).fit(to_logit(p_val), y_val)
    return lambda p: model.predict_proba(to_logit(p))[:, 1]


def bootstrap_auroc(y, p, n=200, seed=0):
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            stats.append(roc_auc_score(y[i], p[i]))
    return np.percentile(stats, [2.5, 97.5])


def audit(model_name, y, p, oracle, groups):
    overall_auc = roc_auc_score(y, p)
    rows = []
    for attribute, values in [("overall", pd.Series(["all"] * len(y)))] + list(groups.items()):
        values = np.asarray(values)
        for level in sorted(set(values)):
            m = values == level
            yy, pp = y[m], p[m]
            if m.sum() < 50 or not 0 < yy.sum() < m.sum():
                continue
            lo, hi = bootstrap_auroc(yy, pp)
            oe, slope = calibration(yy, pp)
            auc = roc_auc_score(yy, pp)
            flags = []
            if overall_auc - auc > FLAG_AUROC_GAP:
                flags.append("lower AUROC")
            if not FLAG_OE[0] <= oe <= FLAG_OE[1]:
                flags.append("miscalibrated")
            rows.append({"model": model_name, "attribute": attribute, "group": level, "n": int(m.sum()),
                         "event_rate": round(yy.mean(), 3), "auroc": round(auc, 3),
                         "auroc_95ci": f"{lo:.3f}-{hi:.3f}", "auprc": round(average_precision_score(yy, pp), 3),
                         "oracle_auroc": round(roc_auc_score(yy, oracle[m]), 3), "obs_exp_ratio": round(oe, 2),
                         "calibration_slope": round(slope, 2), "flag": "; ".join(flags)})
    return pd.DataFrame(rows)


def calibration_plot(y, p, groups, path, model_name):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 0.6], [0, 0.6], ls="--", color="grey", lw=1)
    adi = np.asarray(groups["neighborhood deprivation"])
    for level, color in [("low", "#2471a3"), ("middle", "#7d8c8f"), ("high", "#c0392b")]:
        m = adi == level
        bins = pd.qcut(p[m], 10, duplicates="drop")
        d = pd.DataFrame({"p": p[m], "y": y[m], "bin": bins}).groupby("bin", observed=True).mean()
        ax.plot(d.p, d.y, marker="o", color=color, label=f"{level} deprivation")
    ax.set_xlabel("Predicted risk (decile mean)")
    ax.set_ylabel("Observed event rate")
    ax.set_title(f"{model_name}: calibration by neighborhood deprivation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def read_json(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def model_card(audit_df):
    base = read_json(RESULTS / "baseline_metrics.json") or {}
    gru = read_json(RESULTS / "gru_time_ehr_sdoh_prs_metrics.json")
    contrib = read_json(RESULTS / "contribution_summary.json") or {}
    lr = base.get("logreg_ever_plus_recent_ehr_sdoh_prs")
    perf = []
    if lr:
        perf.append(f"| Logistic regression (ever + recent flags) | {lr['auroc']} | {lr['auprc']} | {lr['brier']} |")
    if gru:
        perf.append(f"| GRU with visit timing | {gru['auroc']} | {gru['auprc']} | {gru['brier']} |")
    if "oracle_true_risk" in base:
        o = base["oracle_true_risk"]
        perf.append(f"| Oracle (true simulated risk; the ceiling) | {o['auroc']} | {o['auprc']} | {o['brier']} |")
    expl = "\n".join(f"| {m} | {v['precision_at_k']} | {v['sign_accuracy']} | {v['spearman_vs_truth']} |"
                     for m, v in contrib.items())
    cols = ["model", "attribute", "group", "n", "event_rate", "auroc", "auroc_95ci", "oracle_auroc",
            "obs_exp_ratio", "calibration_slope", "flag"]
    table = audit_df[cols].to_markdown(index=False)
    flagged = audit_df[audit_df.flag != ""]
    flag_text = ("No subgroup was flagged." if flagged.empty else
                 "Flagged subgroups: " + "; ".join(f"{r.model} / {r.attribute} = {r.group} ({r.flag})"
                                                   for r in flagged.itertuples()) + ".")
    return f"""# Model Card: DeepBiomarker (synthetic-data reimplementation)

*Generated by `python -m deepbiomarker.audit` on {date.today().isoformat()}. Format follows Mitchell et al.,
"Model Cards for Model Reporting" (2019); reporting items are informed by TRIPOD+AI.*

## Model details
- **Audit comparison:** the same logistic regression without social determinants or PRS, to show how omitting
  neighborhood factors affects calibration for different groups.
- **Models:** a GRU over visit embeddings (each visit is the sum of learned code embeddings plus time before the index
  date) combined with patient-level features; a logistic regression on code flags as the comparator.
- **Developer:** Oshin Miranda. Independent reimplementation of the approach in the DeepBiomarker papers; not the
  original study code.
- **License:** MIT.

## Intended use
- **Intended:** demonstrating and testing methods (sequence modeling of EHR, interpretability, subgroup auditing) on
  synthetic data; teaching and research.
- **Out of scope:** any use with real patients or any clinical decision. The model was trained only on simulated data
  and has never been validated on real records.

## Data
Synthetic cohort of 50,000 patients generated by `generate.py`, with known (planted) risk and protective factors in
the visit history, social determinants of health, and a polygenic risk score. Split by patient into 70% training,
15% validation, 15% test. No real patient data is used.

## Performance (test set)
| Model | AUROC | AUPRC | Brier |
|---|---|---|---|
{chr(10).join(perf) if perf else "| (run the baseline and model scripts first) | | | |"}

## Explanation validity
Perturbation-based contributions scored against the planted ground truth:

| Model | Precision@k | Sign accuracy | Spearman vs truth |
|---|---|---|---|
{expl if expl else "| (run `python -m deepbiomarker.contributions` first) | | | |"}

## Subgroup audit (test set)
Calibration: an observed/expected ratio of 1.0 means predicted risk matches observed risk; a calibration slope below
1.0 means predictions are too extreme. The oracle AUROC comes from the true simulated risk and shows how predictable
each group inherently is. Flags: AUROC more than {FLAG_AUROC_GAP} below overall, or observed/expected outside
{FLAG_OE[0]}-{FLAG_OE[1]}.

{table}

{flag_text}

## Ethical considerations
- The outcome (psychiatric hospitalization) is sensitive; a real model could affect access to care if misused.
- Neighborhood deprivation and housing instability can act as proxies for race and income. Using them can improve
  prediction but can also encode existing inequities; any real use would need fairness review with clinicians and
  affected communities, and a decision on whether such features belong in the model at all.
- Explanations show predictive importance, not causation, and should not be read as treatment targets.

## Limitations
- Synthetic data with a simple, additive risk structure: real-world performance, calibration, and subgroup gaps
  would differ.
- Subgroup results in synthetic data reflect the simulation, not real disparities; the value here is the audit
  method, which carries over to real data.
- Calibration was evaluated on the same distribution as training; real deployment would require external validation
  and recalibration.
"""


def main():
    patients, visits = load()
    train, val, test = split_patients(patients)
    vocab = build_vocab(visits, train.patient_id)
    cols = FEATURE_SETS["ehr_sdoh_prs"]
    s_tr, mean, std = static_matrix(train, cols)
    s_te, _, _ = static_matrix(test, cols, mean, std)
    s_va, _, _ = static_matrix(val, cols, mean, std)
    c_tr, t_tr, l_tr = build_sequences(visits, train, vocab)
    c_va, t_va, l_va = build_sequences(visits, val, vocab)
    c_te, t_te, l_te = build_sequences(visits, test, vocab)
    y, oracle, groups = test.label.to_numpy(), test.true_risk.to_numpy(), subgroups(test)

    lr = LogisticRegression(max_iter=3000).fit(
        np.hstack([c_tr.max(axis=1), (c_tr * (t_tr <= 180 / 365)).max(axis=1), s_tr]), train.label)
    preds = {"logreg": lr_from_arrays(lr, c_te, t_te, l_te, s_te)}

    # Comparison: the same model WITHOUT social determinants or PRS (visit history + age only). Leaving out
    # neighborhood factors can make a model systematically under-predict risk for patients in deprived areas.
    age_tr, m_age, sd_age = static_matrix(train, ["age"])
    age_te = static_matrix(test, ["age"], m_age, sd_age)[0]
    lr_ehr = LogisticRegression(max_iter=3000).fit(
        np.hstack([c_tr.max(axis=1), (c_tr * (t_tr <= 180 / 365)).max(axis=1), age_tr]), train.label)
    preds["logreg_without_sdoh"] = lr_from_arrays(lr_ehr, c_te, t_te, l_te, age_te)

    ckpt_path = MODELS / "gru_time_ehr_sdoh_prs.pt"
    if ckpt_path.exists():
        import torch
        from deepbiomarker.model import VisitGRU
        ckpt = torch.load(ckpt_path, weights_only=False)
        gru = VisitGRU(len(ckpt["vocab"]), n_static=len(ckpt["static_cols"]), use_time=ckpt["use_time"])
        gru.load_state_dict(ckpt["state_dict"]); gru.eval()
        @torch.no_grad()
        def gru_predict(c, t, l, s_):
            return np.concatenate([torch.sigmoid(gru(
                torch.tensor(c[i:i + 1024]), torch.tensor(t[i:i + 1024]), torch.tensor(l[i:i + 1024]),
                torch.tensor(s_[i:i + 1024]))).numpy() for i in range(0, len(c), 1024)])
        preds["gru"] = gru_predict(c_te, t_te, l_te, s_te)
        # Recalibration (Platt scaling): fit on the VALIDATION set, apply to test. Early stopping selected the
        # GRU on AUROC, which rewards ranking but ignores whether the probabilities are accurate.
        platt = recalibrator(val.label.to_numpy(), gru_predict(c_va, t_va, l_va, s_va))
        preds["gru_recalibrated"] = platt(preds["gru"])
    else:
        print(f"{ckpt_path} not found: run `python -m deepbiomarker.model` first to include the GRU")

    RESULTS.mkdir(exist_ok=True)
    result = pd.concat([audit(name, y, p, oracle, groups) for name, p in preds.items()], ignore_index=True)
    result.to_csv(RESULTS / "subgroup_audit.csv", index=False)
    main_model = "gru" if "gru" in preds else "logreg"
    calibration_plot(y, preds[main_model], groups, RESULTS / f"calibration_{main_model}.png", main_model)
    Path("MODEL_CARD.md").write_text(model_card(result))
    print(result.drop(columns=["auprc"]).to_string(index=False))
    print("\nWrote results/subgroup_audit.csv, a calibration plot, and MODEL_CARD.md")


if __name__ == "__main__":
    main()

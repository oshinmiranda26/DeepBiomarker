"""Perturbation-based contribution analysis, as in the DeepBiomarker papers.

For each medical code: remove it from every visit of the test patients who have it, and measure how much the
predicted risk drops. For each patient-level feature: shift it down by 1 SD (continuous) or from 1 to 0 (binary).
Contribution = average change in predicted probability; positive means the factor raises risk.

Because the data has planted ground truth, we can score the explanations:
- precision@k: of the top-k factors by |contribution| (k = number of true factors), how many are real?
- sign accuracy: are true risk factors positive and protective factors negative?
- Spearman correlation between contributions and true log-odds across all factors

Run:  python -m deepbiomarker.contributions            (GRU and logistic regression, EHR + SDoH + PRS)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from deepbiomarker.dataset import FEATURE_SETS, build_sequences, build_vocab, load, split_patients, static_matrix
from deepbiomarker.generate import STATIC_EFFECTS, TRUE_EFFECTS

RESULTS, MODELS = Path("results"), Path("models")
BINARY = {"housing_instability", "food_insecurity", "unemployed", "rural"}


def lr_from_arrays(model, codes, time, lengths, static):
    """Logistic regression on 'ever' + 'recent (<=180 days)' flags computed from the visit arrays."""
    ever = codes.max(axis=1)
    recent = (codes * (time <= 180 / 365)).max(axis=1)
    return model.predict_proba(np.hstack([ever, recent, static]))[:, 1]


def contributions(predict, codes, time, lengths, static, vocab, cols, std):
    base = predict(codes, time, lengths, static)
    rows = []
    for code, j in vocab.items():
        has = codes[:, :, j].max(axis=1) > 0
        if has.sum() < 30:
            continue
        c2 = codes[has].copy()
        c2[:, :, j] = 0.0
        delta = base[has] - predict(c2, time[has], lengths[has], static[has])
        rows.append({"feature": code, "contribution": float(delta.mean()), "n_patients": int(has.sum())})
    for k, col in enumerate(cols):
        if col == "age":
            continue
        s2 = static.copy()
        if col in BINARY:
            has = static[:, k] > static[:, k].min()  # patients with the factor present
            s2[has, k] = static[:, k].min()
            delta = base[has] - predict(codes[has], time[has], lengths[has], s2[has])
            n = int(has.sum())
        else:
            s2[:, k] -= 1.0  # standardized units: minus 1 SD
            delta = base - predict(codes, time, lengths, s2)
            n = len(base)
        rows.append({"feature": col, "contribution": float(delta.mean()), "n_patients": n})
    return pd.DataFrame(rows)


def score(df):
    truth = {**TRUE_EFFECTS, **STATIC_EFFECTS}
    df = df.assign(true_log_odds=df.feature.map(truth).fillna(0.0))
    df = df.reindex(df.contribution.abs().sort_values(ascending=False).index).reset_index(drop=True)
    true_set = df.true_log_odds != 0
    k = int(true_set.sum())
    signs = np.sign(df.loc[true_set, "contribution"]) == np.sign(df.loc[true_set, "true_log_odds"])
    return df, {"n_true_factors": k,
                "precision_at_k": round(float(true_set.head(k).mean()), 3),
                "sign_accuracy": round(float(signs.mean()), 3),
                "spearman_vs_truth": round(float(spearmanr(df.contribution, df.true_log_odds).correlation), 3)}


def plot(df, title, path, top=20):
    d = df.head(top).iloc[::-1]
    colors = ["#c0392b" if t > 0 else "#2471a3" if t < 0 else "#aaaaaa" for t in d.true_log_odds]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(d.feature, d.contribution, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Contribution (average change in predicted risk)")
    ax.set_title(f"{title}\nred = planted risk factor, blue = planted protective, grey = noise")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    patients, visits = load()
    train, _, test = split_patients(patients)
    vocab = build_vocab(visits, train.patient_id)
    cols = FEATURE_SETS["ehr_sdoh_prs"]
    s_tr, mean, std = static_matrix(train, cols)
    s_te, _, _ = static_matrix(test, cols, mean, std)
    c_tr, t_tr, l_tr = build_sequences(visits, train, vocab)
    c_te, t_te, l_te = build_sequences(visits, test, vocab)
    RESULTS.mkdir(exist_ok=True)

    lr = LogisticRegression(max_iter=3000).fit(
        np.hstack([c_tr.max(axis=1), (c_tr * (t_tr <= 180 / 365)).max(axis=1), s_tr]), train.label)
    models = {"logreg": lambda c, t, l, s: lr_from_arrays(lr, c, t, l, s)}

    ckpt_path = MODELS / "gru_time_ehr_sdoh_prs.pt"
    if ckpt_path.exists():
        import torch
        from deepbiomarker.model import VisitGRU
        ckpt = torch.load(ckpt_path, weights_only=False)
        gru = VisitGRU(len(ckpt["vocab"]), n_static=len(ckpt["static_cols"]), use_time=ckpt["use_time"])
        gru.load_state_dict(ckpt["state_dict"]); gru.eval()

        @torch.no_grad()
        def gru_predict(c, t, l, s, batch=1024):
            out = [torch.sigmoid(gru(torch.tensor(c[i:i + batch]), torch.tensor(t[i:i + batch]),
                                     torch.tensor(l[i:i + batch]), torch.tensor(s[i:i + batch]))).numpy()
                   for i in range(0, len(c), batch)]
            return np.concatenate(out)
        models["gru"] = gru_predict
    else:
        print(f"{ckpt_path} not found: run `python -m deepbiomarker.model` first to include the GRU")

    summary = {}
    for name, predict in models.items():
        df, metrics = score(contributions(predict, c_te, t_te, l_te, s_te, vocab, cols, std))
        df.to_csv(RESULTS / f"contributions_{name}.csv", index=False)
        plot(df, f"{name}: top contributions on test patients", RESULTS / f"contributions_{name}.png")
        summary[name] = metrics
        print(f"\n{name}: {metrics}\n{df.head(13)[['feature', 'contribution', 'true_log_odds']].to_string(index=False)}")
    (RESULTS / "contribution_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

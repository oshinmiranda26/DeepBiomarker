"""Baseline models: logistic regression on bag-of-codes plus patient-level features.

Visit-history features:
- "ever": did the code appear in the 3-year window (ignores timing)
- "ever + recent": adds a flag for the last 180 days (hand-engineered timing)
Patient-level feature sets: ehr (age), ehr_sdoh (+ social determinants), ehr_sdoh_prs (+ polygenic risk score)

Run:  python -m deepbiomarker.baseline
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from deepbiomarker.dataset import (FEATURE_SETS, bag_of_codes, build_vocab, load, split_patients,
                                   static_matrix)

RESULTS = Path("results")


def evaluate(y, p):
    return {"auroc": round(roc_auc_score(y, p), 4), "auprc": round(average_precision_score(y, p), 4),
            "brier": round(brier_score_loss(y, p), 4)}


def code_features(visits, patients, vocab, with_recent):
    X = bag_of_codes(visits, patients, vocab)
    if with_recent:
        X = np.hstack([X, bag_of_codes(visits, patients, vocab, recent_days=180)])
    return X


def main():
    patients, visits = load()
    train, val, test = split_patients(patients)
    vocab = build_vocab(visits, train.patient_id)
    print(f"train {len(train)} | val {len(val)} | test {len(test)} | vocab {len(vocab)} codes | "
          f"test prevalence {test.label.mean():.1%}")

    results = {"oracle_true_risk": evaluate(test.label, test.true_risk)}  # the best any model could do
    for fs, cols in FEATURE_SETS.items():
        s_tr, mean, std = static_matrix(train, cols)
        s_te, _, _ = static_matrix(test, cols, mean, std)
        for name, recent in [("ever", False), ("ever_plus_recent", True)]:
            X_tr = np.hstack([code_features(visits, train, vocab, recent), s_tr])
            X_te = np.hstack([code_features(visits, test, vocab, recent), s_te])
            model = LogisticRegression(max_iter=3000).fit(X_tr, train.label)
            results[f"logreg_{name}_{fs}"] = evaluate(test.label, model.predict_proba(X_te)[:, 1])

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "baseline_metrics.json").write_text(json.dumps(results, indent=2))
    for k, v in results.items():
        print(f"{k:<36} {v}")


if __name__ == "__main__":
    main()

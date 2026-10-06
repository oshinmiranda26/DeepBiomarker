"""Baseline models: logistic regression on bag-of-codes.

Two versions:
- "ever" features: did the code ever appear in the 3-year window (ignores timing)
- "ever + recent" features: adds a second flag for the last 180 days (hand-engineered timing)
The sequence model later learns timing on its own; these baselines show how much timing matters.

Run:  python -m deepbiomarker.baseline
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from deepbiomarker.dataset import bag_of_codes, build_vocab, load, split_patients

RESULTS = Path("results")


def evaluate(y, p):
    return {"auroc": round(roc_auc_score(y, p), 4), "auprc": round(average_precision_score(y, p), 4),
            "brier": round(brier_score_loss(y, p), 4)}


def features(visits, patients, vocab, with_recent):
    X = bag_of_codes(visits, patients, vocab)
    if with_recent:
        X = np.hstack([X, bag_of_codes(visits, patients, vocab, recent_days=180)])
    age = (patients.age.to_numpy(dtype=np.float32)[:, None] - 50) / 20
    return np.hstack([X, age])


def main():
    patients, visits = load()
    train, val, test = split_patients(patients)
    vocab = build_vocab(visits, train.patient_id)
    print(f"train {len(train)} | val {len(val)} | test {len(test)} | vocab {len(vocab)} codes | "
          f"test prevalence {test.label.mean():.1%}")

    ceiling = evaluate(test.label, test.true_risk)  # the best any model could do on this data
    results = {"oracle_true_risk": ceiling}
    for name, recent in [("logreg_ever", False), ("logreg_ever_plus_recent", True)]:
        model = LogisticRegression(max_iter=2000, C=1.0)
        model.fit(features(visits, train, vocab, recent), train.label)
        results[name] = evaluate(test.label, model.predict_proba(features(visits, test, vocab, recent))[:, 1])

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "baseline_metrics.json").write_text(json.dumps(results, indent=2))
    for k, v in results.items():
        print(f"{k:<26} {v}")


if __name__ == "__main__":
    main()

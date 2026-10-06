"""GRU sequence model over visits, in the style of pytorch_ehr (Zhi lab) and RETAIN-type EHR models.

Each visit becomes a learned embedding: the sum of its codes' embeddings (like Med-BERT and pytorch_ehr,
which represent visits as sets of medical codes), optionally joined with the time before the index date.
A GRU reads the visits in order, and its final state predicts the outcome.

Run:  python -m deepbiomarker.model                              (GRU with timing, EHR + SDoH + PRS)
      python -m deepbiomarker.model --features ehr               (visit history + age only)
      python -m deepbiomarker.model --no-time                    (ablation: visit order only, no timing)
"""
import argparse
import json
import random
import time as clock
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from deepbiomarker.baseline import evaluate
from deepbiomarker.dataset import (FEATURE_SETS, build_sequences, build_vocab, load, split_patients,
                                   static_matrix)

RESULTS, MODELS = Path("results"), Path("models")


class VisitGRU(nn.Module):
    def __init__(self, n_codes, n_static=1, emb_dim=32, hidden=64, use_time=True, dropout=0.2):
        super().__init__()
        self.use_time = use_time
        self.code_emb = nn.Linear(n_codes, emb_dim, bias=False)  # multi-hot x weights = sum of code embeddings
        self.gru = nn.GRU(emb_dim + (1 if use_time else 0), hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden + n_static, 1)  # GRU summary + patient-level features (age, SDoH, PRS)

    def forward(self, codes, time, lengths, static):
        x = self.code_emb(codes)
        if self.use_time:
            x = torch.cat([x, time], dim=-1)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)  # h: final hidden state after each patient's last real visit
        return self.head(torch.cat([self.dropout(h[-1]), static], dim=-1)).squeeze(-1)


def to_tensors(visits, patients, vocab, static):
    codes, time, lengths = build_sequences(visits, patients, vocab)
    return [torch.tensor(a) for a in (codes, time, lengths, static, patients.label.to_numpy(np.float32))]


@torch.no_grad()
def predict(model, tensors, batch=512):
    model.eval()
    codes, time, lengths, age, _ = tensors
    out = [torch.sigmoid(model(codes[i:i + batch], time[i:i + batch], lengths[i:i + batch], age[i:i + batch]))
           for i in range(0, len(codes), batch)]
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-time", action="store_true", help="ablation: drop visit timing, keep order")
    p.add_argument("--features", choices=list(FEATURE_SETS), default="ehr_sdoh_prs")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6, help="early stopping on validation AUROC")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    tag = f"{'gru_no_time' if args.no_time else 'gru_time'}_{args.features}"

    patients, visits = load()
    train, val, test = split_patients(patients)
    vocab = build_vocab(visits, train.patient_id)
    cols = FEATURE_SETS[args.features]
    s_tr, mean, std = static_matrix(train, cols)
    tr = to_tensors(visits, train, vocab, s_tr)
    va = to_tensors(visits, val, vocab, static_matrix(val, cols, mean, std)[0])
    te = to_tensors(visits, test, vocab, static_matrix(test, cols, mean, std)[0])

    model = VisitGRU(len(vocab), n_static=len(cols), use_time=not args.no_time)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state, waited, start = -1.0, None, 0, clock.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(tr[0]))
        for i in range(0, len(order), 128):
            idx = order[i:i + 128]
            logits = model(tr[0][idx], tr[1][idx], tr[2][idx], tr[3][idx])
            loss = loss_fn(logits, tr[4][idx])
            opt.zero_grad(); loss.backward(); opt.step()
        val_auc = evaluate(va[4].numpy(), predict(model, va))["auroc"]
        print(f"epoch {epoch:>2} | val AUROC {val_auc:.4f}")
        if val_auc > best:
            best, waited = val_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= args.patience:
                print(f"early stopping: no improvement for {args.patience} epochs")
                break

    model.load_state_dict(best_state)
    metrics = evaluate(te[4].numpy(), predict(model, te))
    metrics.update({"best_val_auroc": best, "epochs_run": epoch, "train_seconds": round(clock.time() - start, 1)})
    print(f"{tag} test: {metrics}")

    RESULTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    (RESULTS / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "use_time": not args.no_time,
                "static_cols": cols, "static_mean": mean, "static_std": std}, MODELS / f"{tag}.pt")


if __name__ == "__main__":
    main()

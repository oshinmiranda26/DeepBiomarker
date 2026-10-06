"""Load the data, split by patient, and build features.

Splits are by patient (70/15/15, stratified by outcome) so no patient appears in more than one split.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA = Path("data")

FEATURE_SETS = {  # patient-level features added to the visit history
    "ehr": ["age"],
    "ehr_sdoh": ["age", "adi_percentile", "housing_instability", "food_insecurity", "unemployed", "rural"],
    "ehr_sdoh_prs": ["age", "adi_percentile", "housing_instability", "food_insecurity", "unemployed", "rural", "prs"],
}


def static_matrix(patients, cols, mean=None, std=None):
    """Standardized patient-level features; pass training mean/std when transforming val/test."""
    X = patients[cols].to_numpy(dtype=np.float32)
    mean = X.mean(axis=0) if mean is None else mean
    std = X.std(axis=0) + 1e-6 if std is None else std
    return (X - mean) / std, mean, std


def load(data_dir=DATA):
    patients = pd.read_csv(Path(data_dir) / "patients.csv")
    visits = pd.read_csv(Path(data_dir) / "visits.csv")
    visits["codes"] = visits["codes"].str.split(";")
    return patients, visits


def split_patients(patients, seed=42):
    train, rest = train_test_split(patients, test_size=0.30, stratify=patients["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_vocab(visits, patient_ids):
    """Code vocabulary from training patients only (avoids leaking information from val/test)."""
    codes = visits[visits.patient_id.isin(patient_ids)]["codes"].explode().unique()
    return {c: i for i, c in enumerate(sorted(codes))}


def bag_of_codes(visits, patients, vocab, recent_days=None):
    """One row per patient: 1 if the code ever appears (optionally only within the last recent_days)."""
    X = np.zeros((len(patients), len(vocab)), dtype=np.float32)
    row = {pid: i for i, pid in enumerate(patients.patient_id)}
    v = visits[visits.patient_id.isin(row)]
    if recent_days is not None:
        v = v[v.days_before_index <= recent_days]
    for pid, codes in zip(v.patient_id, v.codes):
        for c in codes:
            if c in vocab:
                X[row[pid], vocab[c]] = 1.0
    return X


def build_sequences(visits, patients, vocab, max_visits=20):
    """Arrays for the sequence model, one row per patient, visits in time order (oldest first).

    codes:   (n_patients, max_visits, n_codes) multi-hot codes per visit
    time:    (n_patients, max_visits, 1) years before the index date (0 = index date)
    lengths: (n_patients,) number of real visits; the rest is zero padding
    """
    n, v = len(patients), len(vocab)
    codes = np.zeros((n, max_visits, v), dtype=np.float32)
    time = np.zeros((n, max_visits, 1), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)
    row = {pid: i for i, pid in enumerate(patients.patient_id)}
    sub = visits[visits.patient_id.isin(row)].sort_values(["patient_id", "days_before_index"], ascending=[True, False])
    for pid, group in sub.groupby("patient_id", sort=False):
        i = row[pid]
        group = group.tail(max_visits)  # keep the most recent visits if a patient has more
        for j, (days, cs) in enumerate(zip(group.days_before_index, group.codes)):
            time[i, j, 0] = days / 365.0
            for c in cs:
                if c in vocab:
                    codes[i, j, vocab[c]] = 1.0
        lengths[i] = len(group)
    return codes, time, lengths

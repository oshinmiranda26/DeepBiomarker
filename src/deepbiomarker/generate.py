"""Synthetic longitudinal EHR with planted risk factors (known ground truth).

Each patient has a sequence of visits before an index date. Each visit holds medical codes: diagnoses
(ICD-10), medications, abnormal lab results, and procedures. The outcome is a psychiatric
hospitalization in the 12 months after the index date.

The outcome is generated from a known set of risk and protective factors ("planted biomarkers"), and
recent events (last 180 days) count more than older ones. Because the truth is known, we can check
later whether a model's explanations recover the real risk factors.

Run:  python -m deepbiomarker.generate
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
SEED = 42
N_PATIENTS = 10000
OBS_DAYS = 3 * 365          # observation window before the index date
RECENT_DAYS = 180           # events in this window count fully; older events are down-weighted
OLD_WEIGHT = 0.3

# Background codes with their base prevalence per visit (noise the model must learn to ignore)
BACKGROUND = {
    # diagnoses
    "DX_F33.1_depression": 0.20, "DX_F41.1_anxiety": 0.18, "DX_F43.10_ptsd": 0.08,
    "DX_F10.20_alcohol_dependence": 0.05, "DX_G47.00_insomnia": 0.08, "DX_I10_hypertension": 0.15,
    "DX_E11.9_type2_diabetes": 0.08, "DX_E78.5_hyperlipidemia": 0.12, "DX_J45.9_asthma": 0.05,
    "DX_M54.5_low_back_pain": 0.10, "DX_K21.9_gerd": 0.07, "DX_E66.9_obesity": 0.10,
    # medications
    "RX_sertraline": 0.10, "RX_escitalopram": 0.08, "RX_bupropion": 0.05, "RX_trazodone": 0.06,
    "RX_lorazepam": 0.05, "RX_oxycodone": 0.03, "RX_lithium": 0.02, "RX_quetiapine": 0.03,
    "RX_lisinopril": 0.10, "RX_metformin": 0.06, "RX_atorvastatin": 0.10, "RX_albuterol": 0.04,
    # abnormal labs
    "LAB_crp_high": 0.06, "LAB_vitamin_d_low": 0.08, "LAB_tsh_high": 0.04, "LAB_a1c_high": 0.06,
    "LAB_alt_high": 0.04, "LAB_sodium_low": 0.02, "LAB_wbc_high": 0.04, "LAB_ldl_high": 0.10,
    # procedures
    "PX_psychotherapy_session": 0.10, "PX_ed_visit": 0.04, "PX_primary_care_visit": 0.40,
}

# Planted ground truth: log-odds contribution of having the code in the recent window
TRUE_EFFECTS = {
    "DX_F10.20_alcohol_dependence": 1.8,
    "RX_lorazepam": 1.5,
    "LAB_crp_high": 1.5,
    "PX_ed_visit": 1.2,
    "DX_G47.00_insomnia": 1.0,
    "LAB_vitamin_d_low": 0.9,
    "DX_F43.10_ptsd": 0.9,
    "PX_psychotherapy_session": -1.2,  # protective
    "RX_lithium": -1.0,                # protective
}
INTERCEPT = -3.7

# Social determinants of health (SDoH) and genetics: patient-level features, as in DeepBiomarker2,
# which added community-level SDoH to EMR data. Effects are in log-odds per 1 SD (continuous) or 0 -> 1 (binary).
STATIC_EFFECTS = {
    "adi_percentile": 0.35,        # area deprivation index (neighborhood disadvantage), per SD
    "housing_instability": 0.6,
    "unemployed": 0.4,
    "prs": 0.4,                    # polygenic risk score, per SD
    "food_insecurity": 0.0,        # noise: correlated with disadvantage but no direct effect
    "rural": 0.0,                  # noise
}
ADI_SD = 100 / np.sqrt(12)  # SD of a uniform 0-100 percentile


def _patient_codes(rng, n_visits):
    days = np.sort(rng.integers(1, OBS_DAYS, n_visits))[::-1]  # days before index, most distant first
    visits = []
    for d in days:
        codes = [c for c, p in BACKGROUND.items() if rng.random() < p]
        visits.append((int(d), codes or ["PX_primary_care_visit"]))
    return visits


def static_logit(s):
    """Ground-truth log-odds contribution of SDoH and genetic features."""
    return (STATIC_EFFECTS["adi_percentile"] * (s["adi_percentile"] - 50) / ADI_SD
            + STATIC_EFFECTS["housing_instability"] * s["housing_instability"]
            + STATIC_EFFECTS["unemployed"] * s["unemployed"]
            + STATIC_EFFECTS["prs"] * s["prs"])


def risk_logit(visits, age):
    """Ground-truth log-odds from visits: recent events count fully, older ones at OLD_WEIGHT."""
    weight = {}
    for days_before, codes in visits:
        w = 1.0 if days_before <= RECENT_DAYS else OLD_WEIGHT
        for c in codes:
            if c in TRUE_EFFECTS:
                weight[c] = max(weight.get(c, 0.0), w)  # presence, weighted by most recent occurrence
    return INTERCEPT + 0.01 * (age - 45) + sum(TRUE_EFFECTS[c] * w for c, w in weight.items())


def generate(n_patients=N_PATIENTS, seed=SEED):
    rng = np.random.default_rng(seed)
    patients, rows = [], []
    for i in range(1, n_patients + 1):
        pid = f"P{i:05d}"
        age = int(rng.integers(18, 85))
        visits = _patient_codes(rng, int(rng.integers(3, 21)))
        adi = float(rng.uniform(0, 100))
        disadvantaged = adi / 100  # more deprived neighborhoods -> more housing and food insecurity
        static = {"adi_percentile": round(adi, 1),
                  "housing_instability": int(rng.random() < 0.04 + 0.16 * disadvantaged),
                  "food_insecurity": int(rng.random() < 0.05 + 0.20 * disadvantaged),
                  "unemployed": int(rng.random() < 0.10),
                  "rural": int(rng.random() < 0.25),
                  "prs": round(float(rng.normal()), 3)}
        p = 1 / (1 + np.exp(-(risk_logit(visits, age) + static_logit(static))))
        patients.append({"patient_id": pid, "age": age, "sex": str(rng.choice(["F", "M"])), **static,
                         "true_risk": round(float(p), 4), "label": int(rng.random() < p)})
        for v, (days_before, codes) in enumerate(visits):
            rows.append({"patient_id": pid, "visit_index": v, "days_before_index": days_before,
                         "codes": ";".join(codes)})
    effects = pd.DataFrame([{"feature": c, "true_log_odds": w} for c, w in TRUE_EFFECTS.items()]
                           + [{"feature": c, "true_log_odds": w} for c, w in STATIC_EFFECTS.items()])
    return pd.DataFrame(patients), pd.DataFrame(rows), effects


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patients", type=int, default=N_PATIENTS)
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    patients, visits, effects = generate(n_patients=args.n_patients)
    patients.to_csv(DATA / "patients.csv", index=False)
    visits.to_csv(DATA / "visits.csv", index=False)
    effects.to_csv(DATA / "true_effects.csv", index=False)
    print(f"patients {len(patients)} | visits {len(visits)} | "
          f"outcome prevalence {patients.label.mean():.1%}")


if __name__ == "__main__":
    main()

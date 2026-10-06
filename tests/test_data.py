import numpy as np

from deepbiomarker.dataset import bag_of_codes, build_vocab, split_patients
from deepbiomarker.generate import TRUE_EFFECTS, generate, risk_logit


def small():
    p, v, e = generate(n_patients=300, seed=1)
    v["codes"] = v["codes"].str.split(";")
    return p, v, e


def test_reproducible():
    a, b = generate(n_patients=50, seed=3), generate(n_patients=50, seed=3)
    assert a[1].equals(b[1]) and a[0].equals(b[0])


def test_recent_events_count_more_than_old():
    code = "LAB_crp_high"
    recent = risk_logit([(30, [code])], age=45)
    old = risk_logit([(900, [code])], age=45)
    assert recent > old > risk_logit([(30, ["PX_primary_care_visit"])], age=45)


def test_splits_do_not_share_patients():
    p, v, _ = small()
    tr, va, te = split_patients(p)
    assert not (set(tr.patient_id) & set(va.patient_id) or set(tr.patient_id) & set(te.patient_id))
    assert len(tr) + len(va) + len(te) == len(p)


def test_vocab_from_train_only_and_features_shape():
    p, v, _ = small()
    tr, _, te = split_patients(p)
    vocab = build_vocab(v, tr.patient_id)
    X = bag_of_codes(v, te, vocab)
    assert X.shape == (len(te), len(vocab)) and set(np.unique(X)) <= {0.0, 1.0}


def test_planted_effects_exist_in_vocabulary():
    p, v, _ = small()
    vocab = build_vocab(v, p.patient_id)
    assert set(TRUE_EFFECTS) <= set(vocab)


def test_sequences_are_time_ordered_and_padded():
    from deepbiomarker.dataset import build_sequences
    p, v, _ = small()
    vocab = build_vocab(v, p.patient_id)
    codes, time, lengths = build_sequences(v, p.head(20), vocab)
    assert codes.shape[0] == 20 and lengths.min() >= 3
    for i in range(20):
        t = time[i, :lengths[i], 0]
        assert (np.diff(t) <= 0).all()          # oldest visit first: years-before-index decreases
        assert codes[i, lengths[i]:].sum() == 0  # padding is empty


def test_static_features_standardized_with_train_stats():
    from deepbiomarker.dataset import FEATURE_SETS, static_matrix
    p, _, _ = small()
    tr, _, te = split_patients(p)
    X, mean, std = static_matrix(tr, FEATURE_SETS["ehr_sdoh_prs"])
    assert np.allclose(X.mean(axis=0), 0, atol=1e-4)
    Xte, _, _ = static_matrix(te, FEATURE_SETS["ehr_sdoh_prs"], mean, std)
    assert Xte.shape[1] == X.shape[1]


def test_logreg_contributions_recover_planted_factors():
    from sklearn.linear_model import LogisticRegression
    from deepbiomarker.contributions import contributions, lr_from_arrays, score
    from deepbiomarker.dataset import FEATURE_SETS, build_sequences, static_matrix
    p, v, _ = generate(n_patients=4000, seed=11)
    v["codes"] = v["codes"].str.split(";")
    tr, _, te = split_patients(p)
    vocab = build_vocab(v, tr.patient_id)
    cols = FEATURE_SETS["ehr_sdoh_prs"]
    s_tr, m, sd = static_matrix(tr, cols)
    s_te = static_matrix(te, cols, m, sd)[0]
    c, t, l = build_sequences(v, tr, vocab)
    lr = LogisticRegression(max_iter=3000).fit(np.hstack([c.max(1), (c * (t <= 180 / 365)).max(1), s_tr]), tr.label)
    ce, te_, le = build_sequences(v, te, vocab)
    _, metrics = score(contributions(lambda *a: lr_from_arrays(lr, *a), ce, te_, le, s_te, vocab, cols, sd))
    assert metrics["precision_at_k"] >= 0.6 and metrics["spearman_vs_truth"] > 0.3

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold


def make_model(seed):
    return LGBMClassifier(
        objective="binary",
        n_estimators=250,
        learning_rate=0.025,
        max_depth=3,
        num_leaves=15,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=1.0,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        verbosity=-1,
        n_jobs=-1,
    )


def participant_summary(score, data, seed):
    values = (
        pd.DataFrame({
            "participant_id": data["participant_id"].to_numpy(),
            "score": score,
        })
        .groupby("participant_id")["score"]
        .mean()
        .to_numpy()
    )

    rng = np.random.default_rng(seed)
    bootstrap = np.array([
        rng.choice(values, len(values), replace=True).mean()
        for _ in range(10000)
    ])

    return np.r_[
        values.mean(),
        np.percentile(bootstrap, [2.5, 97.5]),
    ]


def fit_aipw(data, outcome, features, seed):
    data = data.reset_index(drop=True)

    channel_features = pd.get_dummies(
        data["source_channel"],
        prefix="source",
        dtype=float,
    )

    x = np.column_stack([
        data[features].to_numpy(dtype=float),
        channel_features.to_numpy(dtype=float),
    ])
    treatment = data["treatment"].to_numpy(dtype=int)
    outcome_values = data[outcome].to_numpy(dtype=int)
    groups = data["participant_id"].to_numpy()

    propensity = np.full(len(data), np.nan)
    mu1 = np.full(len(data), np.nan)
    mu0 = np.full(len(data), np.nan)

    for fold, (train, test) in enumerate(
        GroupKFold(5).split(x, outcome_values, groups),
        start=1,
    ):
        propensity_model = make_model(seed + fold)
        propensity_model.fit(x[train], treatment[train])
        propensity[test] = propensity_model.predict_proba(x[test])[:, 1]

        treated = train[treatment[train] == 1]
        control = train[treatment[train] == 0]

        mu1[test] = (
            make_model(seed + 100 + fold)
            .fit(x[treated], outcome_values[treated])
            .predict_proba(x[test])[:, 1]
        )
        mu0[test] = (
            make_model(seed + 200 + fold)
            .fit(x[control], outcome_values[control])
            .predict_proba(x[test])[:, 1]
        )

    propensity = np.clip(propensity, 0.05, 0.95)

    score = (
        mu1 - mu0
        + treatment * (outcome_values - mu1) / propensity
        - (1 - treatment)
        * (outcome_values - mu0)
        / (1 - propensity)
    )

    return participant_summary(score, data, seed), score
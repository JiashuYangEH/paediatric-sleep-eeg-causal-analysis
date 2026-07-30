from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from .causal_models import make_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / ".cache"


def load_or_build_permutation_null(
        dynamic_samples,
        seed=42,
        n_permutations=199,
        checkpoint_every=10,
):
    cache_path = CACHE_ROOT / "full_dynamic_label_permutation_v1.csv"
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        completed = set(cached["permutation"].astype(int).tolist())
        expected = set(range(n_permutations))

        if expected.issubset(completed):
            print(f"Loading cached permutations: {cache_path}")
            return (
                cached.loc[cached["permutation"].isin(expected)]
                .sort_values("permutation")
                .reset_index(drop=True)
            )

        permutation_null = cached
        completed_indices = completed
        print(f"Resuming permutation cache: {len(completed_indices)}/{n_permutations}")
    else:
        permutation_null = pd.DataFrame()
        completed_indices = set()

    permutation_data = dynamic_samples.copy().reset_index(drop=True)
    permutation_features = ["run", "time_fraction", "age", "sex_binary", "history_2_10s"]

    channel_dummies = pd.get_dummies(
        permutation_data["source_channel"],
        prefix="source",
        dtype=float,
    )

    X_permutation = np.column_stack([
        permutation_data[permutation_features].to_numpy(dtype=float),
        channel_dummies.to_numpy(dtype=float),
    ])

    participant_groups = permutation_data["participant_id"].to_numpy()
    participant_vector = permutation_data["participant_id"].to_numpy()
    real_treatment = permutation_data["treatment"].to_numpy(dtype=int)
    pair_vector = permutation_data["pair_id"].to_numpy()
    unique_pair_ids = np.unique(pair_vector)

    forward_columns = [f"forward_bin_{i}" for i in range(4)]
    backward_columns = [f"backward_bin_{i}" for i in range(4)]
    outcome_columns = forward_columns + backward_columns
    Y_matrix = permutation_data[outcome_columns].to_numpy(dtype=int)

    # Keep the participant folds identical in every permutation.
    group_cv = GroupKFold(n_splits=5)
    fixed_folds = list(group_cv.split(X_permutation, real_treatment, participant_groups))

    def fit_complete_dynamic_aipw(treatment_vector):
        """
        Refit propensity and all eight outcome models.

        Returns four participant-weighted directional effects:
            forward window i - backward window i
        """
        n_observations = len(permutation_data)
        n_outcomes = len(outcome_columns)

        propensity = np.full(n_observations, np.nan)
        mu1 = np.full((n_observations, n_outcomes), np.nan)
        mu0 = np.full((n_observations, n_outcomes), np.nan)

        for fold_number, (train_indices, test_indices) in enumerate(fixed_folds, start=1):
            # Propensity model
            propensity_model = make_model(seed + 12000 + fold_number)
            propensity_model.fit(X_permutation[train_indices], treatment_vector[train_indices])

            propensity[test_indices] = propensity_model.predict_proba(
                X_permutation[test_indices]
            )[:, 1]

            treated_train = train_indices[treatment_vector[train_indices] == 1]
            control_train = train_indices[treatment_vector[train_indices] == 0]

            # Eight outcome models
            for outcome_index in range(n_outcomes):
                current_outcome = Y_matrix[:, outcome_index]

                treated_model = make_model(seed + 13000 + 100 * outcome_index + fold_number)
                control_model = make_model(seed + 14000 + 100 * outcome_index + fold_number)

                treated_model.fit(X_permutation[treated_train], current_outcome[treated_train])
                control_model.fit(X_permutation[control_train], current_outcome[control_train])

                mu1[test_indices, outcome_index] = treated_model.predict_proba(
                    X_permutation[test_indices]
                )[:, 1]
                mu0[test_indices, outcome_index] = control_model.predict_proba(
                    X_permutation[test_indices]
                )[:, 1]

        assert np.isfinite(propensity).all()
        assert np.isfinite(mu1).all()
        assert np.isfinite(mu0).all()

        clipped_propensity = np.clip(propensity, 0.05, 0.95)
        A = treatment_vector[:, None]

        # Cross-fitted doubly robust AIPW scores.
        aipw_scores = (
                mu1 - mu0
                + A * (Y_matrix - mu1) / clipped_propensity[:, None]
                - (1 - A) * (Y_matrix - mu0) / (1 - clipped_propensity[:, None])
        )

        score_table = pd.DataFrame(aipw_scores, columns=outcome_columns)
        score_table["participant_id"] = participant_vector

        # Equal weighting of the 30 participants.
        participant_effects = score_table.groupby("participant_id")[outcome_columns].mean()
        average_effects = participant_effects.mean(axis=0)

        forward_effects = average_effects[forward_columns].to_numpy()
        backward_effects = average_effects[backward_columns].to_numpy()
        directional_effects = forward_effects - backward_effects

        return {
            "directional_effects": directional_effects,
            "forward_effects": forward_effects,
            "backward_effects": backward_effects,
            "propensity_min": propensity.min(),
            "propensity_max": propensity.max(),
        }

    new_permutation_rows = []
    permutation_start = time.time()

    for permutation_index in range(n_permutations):
        if permutation_index in completed_indices:
            continue

        # Independent deterministic RNG for every permutation,
        # allowing an interrupted run to resume exactly.
        permutation_rng = np.random.default_rng(seed + 20000 + permutation_index)

        # Each matched pair has exactly two observations:
        # one source-HFO time and one matched control time.
        # A value of 1 means their treatment labels are exchanged.
        swap_pair = dict(
            zip(unique_pair_ids, permutation_rng.integers(0, 2, size=len(unique_pair_ids)))
        )

        swap_row = np.array([swap_pair[pair_id] for pair_id in pair_vector])

        permuted_treatment = np.where(swap_row == 1, 1 - real_treatment, real_treatment).astype(int)
        permuted_fit = fit_complete_dynamic_aipw(permuted_treatment)
        permuted_directional = permuted_fit["directional_effects"]

        new_permutation_rows.append({
            "permutation": permutation_index,
            "effect_25_144ms": permuted_directional[0],
            "effect_144_262ms": permuted_directional[1],
            "effect_262_381ms": permuted_directional[2],
            "effect_381_500ms": permuted_directional[3],
            "max_absolute_effect": np.max(np.abs(permuted_directional)),
            "max_positive_effect": np.max(permuted_directional),
            "propensity_min": permuted_fit["propensity_min"],
            "propensity_max": permuted_fit["propensity_max"],
        })

        completed_now = len(completed_indices) + len(new_permutation_rows)
        print(f"\rFull label permutation {completed_now}/{n_permutations}", end="")

        # Save regularly so a long run is recoverable.
        if len(new_permutation_rows) % checkpoint_every == 0:
            updated_null = pd.concat(
                [permutation_null, pd.DataFrame(new_permutation_rows)],
                ignore_index=True,
            )

            updated_null = (
                updated_null.drop_duplicates(subset="permutation", keep="last")
                .sort_values("permutation")
            )

            updated_null.to_csv(cache_path, index=False)

    print()

    # Final save.
    permutation_null = pd.concat(
        [permutation_null, pd.DataFrame(new_permutation_rows)],
        ignore_index=True,
    )

    permutation_null = (
        permutation_null.drop_duplicates(subset="permutation", keep="last")
        .sort_values("permutation")
        .reset_index(drop=True)
    )

    permutation_null.to_csv(cache_path, index=False)

    return permutation_null
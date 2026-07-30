from pathlib import Path
import re
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / ".cache"

BURST_MERGE_SECONDS = 0.025
MIN_PROPAGATION_LAG = 0.025
MAX_PROPAGATION_LAG = 0.500
CONTROL_MIN_OFFSET = 2.0
CONTROL_MAX_OFFSET = 30.0
RUN_DURATION_SECONDS = 300.0


def get_accepted_hfo(events):
    accepted_mask = pd.to_numeric(events["EvPassRejection"], errors="coerce").eq(1)
    return events.loc[accepted_mask].copy()


def channel_electrodes(channel):
    return frozenset(str(channel).split("-"))


def channels_are_spatially_separate(source_channel, target_channel):
    return channel_electrodes(source_channel).isdisjoint(channel_electrodes(target_channel))


def load_run_bursts(events_path):
    events = pd.read_csv(events_path, sep="\t")
    accepted = get_accepted_hfo(events).copy()

    if accepted.empty:
        return pd.DataFrame(columns=["burst_time", "channels", "n_channels"])

    channels_path = events_path.with_name(events_path.name.replace("_events.tsv", "_channels.tsv"))
    channel_table = pd.read_csv(channels_path, sep="\t")
    event_sfreq = float(channel_table["sampling_frequency"].iloc[0])

    accepted["burst_time"] = (
        pd.to_numeric(accepted["indStart"], errors="coerce")
        + pd.to_numeric(accepted["indStop"], errors="coerce")
    ) / (2 * event_sfreq)

    accepted = (
        accepted.dropna(subset=["burst_time", "strChannelName"])
        .sort_values("burst_time")
        .reset_index(drop=True)
    )

    event_times = accepted["burst_time"].to_numpy()

    # Single-linkage temporal clustering.
    new_burst = np.r_[True, np.diff(event_times) > BURST_MERGE_SECONDS]
    accepted["burst_id"] = np.cumsum(new_burst) - 1

    bursts = accepted.groupby("burst_id", as_index=False).agg(
        burst_time=("burst_time", "median"),
        channels=("strChannelName", lambda values: tuple(sorted(set(values)))),
    )

    bursts["n_channels"] = bursts["channels"].map(len)

    return bursts


def has_distant_target_after(burst_times, burst_channels, source_time, source_channel):
    """Target burst 25–500 ms after the source."""
    start = source_time + MIN_PROPAGATION_LAG
    stop = source_time + MAX_PROPAGATION_LAG

    left = np.searchsorted(burst_times, start, side="left")
    right = np.searchsorted(burst_times, stop, side="right")

    for target_channels in burst_channels[left:right]:
        for target_channel in target_channels:
            if channels_are_spatially_separate(source_channel, target_channel):
                return 1

    return 0


def has_distant_target_before(burst_times, burst_channels, source_time, source_channel):
    """
    was there an eligible target 25–500 ms before the source?
    """
    start = source_time - MAX_PROPAGATION_LAG
    stop = source_time - MIN_PROPAGATION_LAG

    left = np.searchsorted(burst_times, start, side="left")
    right = np.searchsorted(burst_times, stop, side="right")

    for target_channels in burst_channels[left:right]:
        for target_channel in target_channels:
            if channels_are_spatially_separate(source_channel, target_channel):
                return 1

    return 0


def recent_burst_count(burst_times, reference_time):
    """
    Number of bursts in the preceding 500–25 ms.
    """
    start = reference_time - MAX_PROPAGATION_LAG
    stop = reference_time - MIN_PROPAGATION_LAG

    left = np.searchsorted(burst_times, start, side="left")
    right = np.searchsorted(burst_times, stop, side="right")

    return int(right - left)


def sample_matched_control_time(burst_times, source_time, rng, max_attempts=500):
    """
    Select a nearby time from the same run that:

    1. is 2–30 seconds from the source;
    2. is not itself within 25 ms of a burst;
    3. has the same preceding 500-ms burst count;
    4. has enough room for the outcome window.
    """
    source_history = recent_burst_count(burst_times, source_time)

    for _ in range(max_attempts):
        direction = rng.choice([-1.0, 1.0])
        distance = rng.uniform(CONTROL_MIN_OFFSET, CONTROL_MAX_OFFSET)

        candidate_time = source_time + direction * distance

        if not (MAX_PROPAGATION_LAG < candidate_time < RUN_DURATION_SECONDS - MAX_PROPAGATION_LAG):
            continue

        nearest_index = np.searchsorted(burst_times, candidate_time)
        nearby_indices = [nearest_index - 1, nearest_index]

        too_close_to_event = any(
            0 <= index < len(burst_times)
            and abs(burst_times[index] - candidate_time) <= BURST_MERGE_SECONDS
            for index in nearby_indices
        )

        if too_close_to_event:
            continue

        candidate_history = recent_burst_count(burst_times, candidate_time)

        if candidate_history != source_history:
            continue

        return candidate_time

    return np.nan


def load_or_build_hfo_event_study(data_root, seed=42):
    data_root = Path(data_root)
    derivatives_root = data_root / "derivatives"
    cache_path = CACHE_ROOT / "hfo_burst_event_study_v1.csv"

    if cache_path.exists():
        print(f"Loading cached event study: {cache_path}")
        return pd.read_csv(cache_path)

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Loading propagation cache: {cache_path}")
        return pd.read_csv(cache_path)

    rng = np.random.default_rng(seed)

    event_paths = sorted(derivatives_root.glob("sub-*/ses-01/eeg/*_run-*_events.tsv"))

    matched_rows = []
    total_channel_events = 0
    total_bursts = 0
    total_multichannel_bursts = 0
    total_single_channel_sources = 0

    pair_id = 0

    for file_index, events_path in enumerate(event_paths, start=1):
        match = re.search(r"(sub-\d+).*?_run-(\d+)_events\.tsv$", events_path.name)
        if match is None:
            continue

        participant_id = match.group(1)
        run = int(match.group(2))

        raw_events = pd.read_csv(events_path, sep="\t")
        accepted = get_accepted_hfo(raw_events)
        total_channel_events += len(accepted)

        bursts = load_run_bursts(events_path)

        if bursts.empty:
            continue

        total_bursts += len(bursts)
        total_multichannel_bursts += int(bursts["n_channels"].gt(1).sum())

        burst_times = bursts["burst_time"].to_numpy(dtype=float)
        burst_channels = bursts["channels"].tolist()

        # Only single-channel bursts are used as source events.
        source_mask = (
            bursts["n_channels"].eq(1)
            & bursts["burst_time"].gt(MAX_PROPAGATION_LAG)
            & bursts["burst_time"].lt(RUN_DURATION_SECONDS - MAX_PROPAGATION_LAG)
        )
        source_bursts = bursts.loc[source_mask]
        total_single_channel_sources += len(source_bursts)

        for _, source in source_bursts.iterrows():
            source_time = float(source["burst_time"])
            source_channel = source["channels"][0]
            control_time = sample_matched_control_time(burst_times, source_time, rng)

            if not np.isfinite(control_time):
                continue

            exposed_outcome = has_distant_target_after(burst_times, burst_channels, source_time, source_channel)
            control_outcome = has_distant_target_after(burst_times, burst_channels, control_time, source_channel)
            pre_source_outcome = has_distant_target_before(burst_times, burst_channels, source_time, source_channel)

            matched_rows.append({
                "pair_id": pair_id,
                "participant_id": participant_id,
                "run": run,
                "source_channel": source_channel,
                "source_time": source_time,
                "control_time": control_time,
                "recent_burst_count": recent_burst_count(burst_times, source_time),
                "exposed_post_outcome": exposed_outcome,
                "control_post_outcome": control_outcome,
                "source_pre_outcome": pre_source_outcome,
            })

            pair_id += 1

        print(f"\rProcessing event file {file_index}/{len(event_paths)}", end="")

    print()

    event_study = pd.DataFrame(matched_rows)
    event_study.to_csv(cache_path, index=False)

    return event_study


def count_bursts_in_history(burst_times, reference_time, earliest_lag, latest_lag):
    """
    Count bursts between reference_time - latest_lag and reference_time - earliest_lag
    """
    start = reference_time - latest_lag
    stop = reference_time - earliest_lag

    left = np.searchsorted(burst_times, start, side="left")
    right = np.searchsorted(burst_times, stop, side="right")

    return int(right - left)


def count_source_channel_history(burst_times, burst_channels, reference_time, source_channel, history_seconds=10.0):
    """Previous bursts involving the same source channel."""
    start = reference_time - history_seconds
    stop = reference_time - BURST_MERGE_SECONDS

    left = np.searchsorted(burst_times, start, side="left")
    right = np.searchsorted(burst_times, stop, side="right")
    count = 0

    for channels_at_burst in burst_channels[left:right]:
        if source_channel in channels_at_burst:
            count += 1

    return count


def count_recent_active_channels(burst_times, burst_channels, reference_time, history_seconds=2.0):
    """Number of distinct channels active before treatment."""
    start = reference_time - history_seconds
    stop = reference_time - BURST_MERGE_SECONDS

    left = np.searchsorted(burst_times, start, side="left")
    right = np.searchsorted(burst_times, stop, side="right")

    active_channels = set()

    for channels_at_burst in burst_channels[left:right]:
        active_channels.update(channels_at_burst)

    return len(active_channels)


def make_pre_treatment_features(burst_times, burst_channels, reference_time, source_channel):
    """
    All features end before treatment time.
    """
    return {
        "history_25_100ms": count_bursts_in_history(
            burst_times, reference_time, earliest_lag=0.025, latest_lag=0.100
        ),
        "history_100_500ms": count_bursts_in_history(
            burst_times, reference_time, earliest_lag=0.100, latest_lag=0.500
        ),
        "history_500ms_2s": count_bursts_in_history(
            burst_times, reference_time, earliest_lag=0.500, latest_lag=2.0
        ),
        "history_2_10s": count_bursts_in_history(
            burst_times, reference_time, earliest_lag=2.0, latest_lag=10.0
        ),
        "source_channel_history_10s": count_source_channel_history(
            burst_times, burst_channels, reference_time, source_channel, history_seconds=10.0
        ),
        "active_channels_history_2s": count_recent_active_channels(
            burst_times, burst_channels, reference_time, history_seconds=2.0
        ),
        "distant_HFO_before": has_distant_target_before(
            burst_times, burst_channels, reference_time, source_channel
        ),
        "time_fraction": reference_time / RUN_DURATION_SECONDS,
    }


def load_or_build_causal_samples(data_root, event_study):
    data_root = Path(data_root)
    derivatives_root = data_root / "derivatives"
    cache_path = CACHE_ROOT / "hfo_propagation_causal_samples_v1.csv"

    if cache_path.exists():
        print(f"Loading cached causal samples: {cache_path}")
        return pd.read_csv(cache_path)

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Loading Causal AI samples: {cache_path}")
        return pd.read_csv(cache_path)

    rows = []
    grouped_pairs = event_study.groupby(["participant_id", "run"], sort=False)

    for group_index, ((participant_id, run), pair_rows) in enumerate(grouped_pairs, start=1):
        events_path = derivatives_root / participant_id / "ses-01" / "eeg" / (
            f"{participant_id}_ses-01_task-hfo_run-{int(run):02d}_events.tsv"
        )

        bursts = load_run_bursts(events_path)

        if bursts.empty:
            continue

        burst_times = bursts["burst_time"].to_numpy(dtype=float)
        burst_channels = bursts["channels"].tolist()

        for _, pair in pair_rows.iterrows():
            source_channel = pair["source_channel"]

            observation_times = [
                (1, float(pair["source_time"])),
                (0, float(pair["control_time"])),
            ]

            for treatment, reference_time in observation_times:
                history = make_pre_treatment_features(burst_times, burst_channels, reference_time, source_channel)
                outcome = has_distant_target_after(burst_times, burst_channels, reference_time, source_channel)

                rows.append({
                    "pair_id": int(pair["pair_id"]),
                    "participant_id": participant_id,
                    "run": int(run),
                    "source_channel": source_channel,
                    "reference_time": reference_time,
                    "treatment": treatment,
                    "outcome": outcome,
                    **history,
                })

        print(f"\rBuilding causal samples {group_index}/{len(grouped_pairs)}", end="")

    print()

    causal_samples = pd.DataFrame(rows)

    participants = pd.read_csv(data_root / "participants.tsv", sep="\t")
    participants["age"] = pd.to_numeric(participants["age"], errors="coerce")
    participants["sex_binary"] = participants["sex"].map({"f": 0.0, "m": 1.0})

    causal_samples = causal_samples.merge(
        participants[["participant_id", "age", "sex_binary"]],
        on="participant_id",
        how="left",
        validate="many_to_one",
    )

    causal_samples.to_csv(cache_path, index=False)
    print(f"Saved: {cache_path}")

    return causal_samples


def spatially_separate(channel_a, channel_b):
    return set(str(channel_a).split("-")).isdisjoint(str(channel_b).split("-"))


def window_outcome(burst_times, burst_channels, reference_time, source_channel, window):
    left = np.searchsorted(burst_times, reference_time + window[0], side="left")
    right = np.searchsorted(burst_times, reference_time + window[1], side="right")

    return int(any(
        spatially_separate(source_channel, target)
        for targets in burst_channels[left:right]
        for target in targets
    ))
"""Download the subset of OpenNeuro ds003555 required by this project."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


DATASET_ID = "ds003555"
EXPECTED_DOI = "10.18112/openneuro.ds003555.v1.0.1"
S3_ROOT = f"s3://openneuro.org/{DATASET_ID}"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / DATASET_ID

TOP_LEVEL_FILES = [
    ".bidsignore",
    ".gitattributes",
    "CHANGES",
    "dataset_description.json",
    "participants.json",
    "participants.tsv",
    "README",
    "task-hfo_eeg.json",
]


def run_aws(*arguments: str) -> None:
    command = ["aws", *arguments]

    print("$", " ".join(command))

    subprocess.run(command, check=True)


def validate_download() -> None:
    required_files = [
        DATA_ROOT / "dataset_description.json",
        DATA_ROOT / "participants.tsv",
        DATA_ROOT / "sub-01" / "ses-01" / "eeg" / "sub-01_ses-01_task-hfo_eeg.edf",
        DATA_ROOT / "derivatives" / "sub-01" / "ses-01" / "eeg" / "DataIntervals.tsv",
    ]

    missing = [path for path in required_files if not path.exists()]

    if missing:
        raise RuntimeError(
            "The dataset download is incomplete:\n"
            + "\n".join(map(str, missing))
        )

    description = json.loads(
        (DATA_ROOT / "dataset_description.json").read_text(encoding="utf-8")
    )

    actual_doi = description.get("DatasetDOI")

    if actual_doi != EXPECTED_DOI:
        raise RuntimeError(
            "Unexpected ds003555 snapshot.\n"
            f"Expected DOI: {EXPECTED_DOI}\n"
            f"Downloaded DOI: {actual_doi}"
        )

    event_files = list((DATA_ROOT / "derivatives").rglob("*_events.tsv"))
    channel_files = list((DATA_ROOT / "derivatives").rglob("*_channels.tsv"))
    interval_files = list((DATA_ROOT / "derivatives").rglob("DataIntervals.tsv"))

    print()
    print("Dataset validation passed:")
    print(f"  DOI: {actual_doi}")
    print(f"  event tables: {len(event_files)}")
    print(f"  channel tables: {len(channel_files)}")
    print(f"  interval tables: {len(interval_files)}")
    print(f"  location: {DATA_ROOT}")


def main() -> None:
    aws = shutil.which("aws")

    if aws is None:
        raise SystemExit(
            "AWS CLI was not found.\n"
            "Install it from https://aws.amazon.com/cli/ "
            "and run this command again."
        )

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    print("Downloading the analysis subset of OpenNeuro ds003555.")
    print("This downloads approximately 0.5 GB instead of the complete dataset.")
    print(f"Destination: {DATA_ROOT}")
    print()

    # Raw sub-01 recording required for PSD and waveform plots.
    run_aws(
        "s3",
        "sync",
        "--no-sign-request",
        f"{S3_ROOT}/sub-01",
        str(DATA_ROOT / "sub-01"),
    )

    # Only derivative tables and descriptions are needed.
    # The large derivative EDF files are deliberately excluded.
    run_aws(
        "s3",
        "sync",
        "--no-sign-request",
        f"{S3_ROOT}/derivatives",
        str(DATA_ROOT / "derivatives"),
        "--exclude",
        "*",
        "--include",
        "*.tsv",
        "--include",
        "*.json",
    )

    # Download the small BIDS metadata files at dataset root.
    for filename in TOP_LEVEL_FILES:
        run_aws(
            "s3",
            "cp",
            "--no-sign-request",
            f"{S3_ROOT}/{filename}",
            str(DATA_ROOT / filename),
        )

    validate_download()


if __name__ == "__main__":
    main()
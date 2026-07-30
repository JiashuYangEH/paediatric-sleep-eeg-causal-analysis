# Paediatric Sleep EEG Causal Analysis

Exploratory causal analysis of paediatric sleep EEG and high-frequency
oscillation (HFO) dynamics using the public
[OpenNeuro ds003555](https://openneuro.org/datasets/ds003555/versions/1.0.1)
dataset.

This repository is a preparatory study for research on causal AI and sleep EEG
in children with neurodevelopmental conditions. It asks whether an HFO on one
scalp EEG channel is followed by an HFO on a spatially non-overlapping channel
more often than a matched non-HFO reference time, after adjustment for recent
HFO activity and recording context.

> **Interpretation:** the analysis provides evidence of short-lag directional
> asymmetry compatible with HFO propagation.

## Analysis overview

The main workflow is contained in
[`causal_ai_epilepsy.ipynb`](causal_ai_epilepsy.ipynb) and includes:

- loading and visualising paediatric sleep EEG with MNE-Python;
- comparing spectral characteristics of N3 and REM sleep;
- constructing matched HFO and non-HFO reference pairs within the same
  participant, N3 run, and source channel;
- defining the primary outcome as a spatially non-overlapping HFO occurring
  25–500 ms after the reference time;
- participant-grouped five-fold cross-fitting with LightGBM nuisance models;
- augmented inverse probability weighting (AIPW);
- participant-level bootstrap uncertainty estimates;
- backward-placebo and quiet-history sensitivity analyses;
- a dynamic effect curve across the 25–500 ms post-reference interval; and
- 199 full-refit, within-pair label permutations.

The target estimand is the participant-average adjusted risk difference. The
current analysis finds an increase of approximately six percentage points in
the probability of a subsequent spatially non-overlapping HFO. The largest
estimated difference occurs in the earliest 25–144 ms window, and the full
permutation analysis gives a two-sided empirical p-value of 0.005.

## Repository structure

```text
.
├── causal_ai_epilepsy.ipynb   # Main analysis and narrative
├── cache/                     # Reusable intermediate analysis tables
├── data/
│   └── ds003555/              # Downloaded data (not distributed by this repo)
├── scripts/
│   └── download_data.py       # Downloads the required ds003555 subset
└── src/
    ├── causal_models.py       # Cross-fitting, AIPW, and bootstrap summaries
    ├── hfo_events.py          # Event construction and cache handling
    ├── permutation.py         # Resumable full-refit permutation analysis
    └── visualization.py       # Analysis diagrams
```

## Installation

Python 3.9 or later is recommended. Clone the repository and create an isolated
environment:

```bash
git clone https://github.com/JiashuYangEH/paediatric-sleep-eeg-causal-analysis.git
cd paediatric-sleep-eeg-causal-analysis
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

or on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install jupyterlab ipykernel mne edfio numpy pandas matplotlib scipy scikit-learn lightgbm
```

The data-download helper also requires the
[AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
to be installed and available as `aws` on the command line. No AWS account is
required for the public OpenNeuro bucket.

## Download the data

From the repository root, run:

```bash
python scripts/download_data.py
```

The script downloads approximately 0.5 GB into `data/ds003555/`. It retrieves:

- the complete raw `sub-01/` directory;
- derivative JSON and TSV annotation files, excluding derivative EDF copies;
  and
- the root BIDS metadata required to identify and validate the dataset.

The downloader checks the dataset DOI and expected paths after transfer. It
does not download the complete ds003555 dataset.

If ds003555 is already stored elsewhere, set `DS003555_ROOT` before launching
Jupyter:

```bash
export DS003555_ROOT=/absolute/path/to/ds003555
```

Windows PowerShell equivalent:

```powershell
$env:DS003555_ROOT = "D:\absolute\path\to\ds003555"
```

## Run the analysis

Start Jupyter from the repository root so that project-relative paths and
imports resolve correctly:

```bash
jupyter lab causal_ai_epilepsy.ipynb
```

Run the notebook from top to bottom. The first cells validate the dataset
location and report any missing files.

## Cache behaviour

The repository includes three small, versioned CSV caches:

- `cache/hfo_burst_event_study_v1.csv`;
- `cache/hfo_propagation_causal_samples_v1.csv`; and
- `cache/full_dynamic_label_permutation_v1.csv`.

The notebook uses a **load-or-build** workflow. If a compatible cache exists,
it is loaded. Otherwise, the corresponding table is generated from ds003555
and saved to `cache/` for later runs. The permutation routine writes
checkpoints and can resume an incomplete run; rebuilding all 199 full-refit
permutations can take substantially longer than loading the included result.

To regenerate an intermediate result, remove only its corresponding cache file
and rerun the relevant notebook cells. The raw data are still required for the
earlier inspection and signal-processing sections of the notebook.

## Reproducibility safeguards

- Matching is performed within participant, N3 run, and source channel.
- Covariates are calculated only from information available before each
  reference time.
- Cross-fitting is grouped by participant to avoid participant leakage between
  training and validation folds.
- The estimand gives participants equal weight.
- Propensity scores are clipped to the interval `[0.05, 0.95]`.
- Uncertainty is estimated with 10,000 participant-level bootstrap samples.
- The permutation null is generated by swapping labels within matched pairs and
  refitting all nuisance models.

Random seeds used by the notebook and permutation code make repeated runs
reproducible, subject to differences in package versions and numerical
libraries.

## Limitations

1. ds003555 contains children and adolescents with epilepsy and selected N3
   intervals.
2. Matching and measured-history adjustment reduce time-varying or unmeasured confounding.
3. Non-overlapping bipolar channel labels provide spatial separation at the
   sensor level. Scalp volume conduction remains a concern.
4. The non-null backward placebo indicates that the temporal structure is not
   perfectly captured.

## Data and citation

The EEG data are not redistributed by this repository. They are available from
OpenNeuro under the dataset's own terms:

> Cserpan, D., Boran, E., Rosch, R., Lo Biundo, S. P., Ramantani, G., &
> Sarnthein, J. (2021). *Dataset of EEG recordings of pediatric patients with
> epilepsy based on the 10-20 system* (Version 1.0.1) [Data set]. OpenNeuro.
> https://doi.org/10.18112/openneuro.ds003555.v1.0.1

Scientific context for the HFO-sequence analysis:

> Cai, Z., Jiang, X., Bagić, A., Worrell, G. A., Richardson, M., & He, B.
> (2024). *Spontaneous HFO sequences reveal propagation pathways for precise
> delineation of epileptogenic networks*. bioRxiv.
> https://doi.org/10.1101/2024.05.02.592202

If you use this repository, please cite the ds003555 dataset and link to this
repository. A software citation will be added when a citable release is
published.

## Intended use

This project is for research and education.
```

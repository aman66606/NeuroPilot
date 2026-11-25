# data/
- `neuropilot_synthetic_eeg_dataset.csv`: synthetic labelled feature dataset (importable in Dataset Lab).
- Real datasets are NOT included (check each licence). To reproduce the real-EEG results:
  - Bonn EEG (seizure detection): download the five folders Z, O, N, F, S into `data/bonn/`, then
    `python scripts/evaluate_bonn.py data/bonn --out docs/bonn_results.md`  (cite Andrzejak et al., 2001).
  - PhysioNet motor imagery (BCI): `scripts/prepare_physionet.py` + `scripts/evaluate_real.py` (see README).

# training/ — SFT data generation only

This folder now contains only `generate_sft_data.py` and its launcher —
the SFT data generator that produces search trajectories for warm-starting
the RL policy.

## What changed

The Tinker-based training scripts that used to live here have been moved
to `training_local_backup/` and replaced by a fully local pipeline in
`training_local/`:

| Old (Tinker-based, removed) | New (local, in training_local/) |
|---|---|
| `train_rl.py` | `training_local/train_rl.py` |
| `train_sft.py` | `training_local/train_sft.py` |
| `launch_rl.sh` | `training_local/launch_rl.sh` |
| `launch_sft_training.sh` | `training_local/launch_sft.sh` |

## Files

- `generate_sft_data.py` — uses GPT-5.4 to generate synthetic search
  trajectories on BrowseComp+, SEC, patents, web corpora. Output: JSON
  files in `tmp/sft_data/`.
- `launch_sft_generation.sh` — bash launcher with sensible defaults.

## Usage

```bash
# Generate ~50 SFT trajectories (small smoke)
bash training/launch_sft_generation.sh

# Production scale
NUM_QUERIES=200 DATASETS=browsecompplus,sec,patents,web \
    bash training/launch_sft_generation.sh
```

## Then what?

After SFT data is generated, run SFT warm-start and then RL training
via `training_local/`:

```bash
bash training_local/launch_sft.sh
INIT_FROM_CHECKPOINT=outputs/sft_runs/<run>/final \
    bash training_local/launch_rl.sh
```

See `training_local/README.md` and `docs/training_guide.md` for full details.

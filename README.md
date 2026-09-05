# Toy Agility Forge Digital Twin

This repository is a work-in-progress implementation of a point-cloud
surrogate workflow for billet forging. The current public files cover the data
preparation boundary before ForgeNet: analytic billet sampling, conversion of
raw JAX-FORGE SQLite records into transition shards, and lazy loading of those
shards for machine learning.

The intended surrogate contract is

$$
\widehat{\Delta X}_t=f_\theta(X_t,c_t),
\qquad
\widehat X_{t+1}=X_t+\widehat{\Delta X}_t,
$$

where `X_t` is an aligned surface point cloud and `c_t` is one scalar
compression action. Pose is not a neural-network input.

> **Project status:** The public repository does not yet contain ForgeNet,
> a supported training entry point, MPC, a checkpoint, or the high-fidelity
> dataset. Those components will be added only after they are reviewed.

## Current files

| File | Responsibility |
|---|---|
| [`sampling.py`](sampling.py) | Generate an analytic cylindrical surface point cloud with a requested total number of points. |
| [`sqlite_to_forgenet_shards.py`](sqlite_to_forgenet_shards.py) | Convert consecutive JAX-FORGE SQLite strike endpoints into smaller ForgeNet-style NPZ transition shards. |
| [`dataset.py`](dataset.py) | Validate and lazily load transition shards, construct PyTorch samples, and split complete trajectories into train, validation, and test sets. |

## Current data flow

```text
Raw JAX-FORGE SQLite database
        |
        v
sqlite_to_forgenet_shards.py
        |
        v
shard_00000.npz, shard_00001.npz, ...
        |
        v
dataset.py
        |
        v
train / validation / test datasets
        |
        v
ForgeNet                         (not yet published)
```

`sampling.py` is a separate analytic billet utility. Its generated points are
not automatically the same material-point identities as the JAX-FORGE surface
vertices stored in the high-fidelity shards.

## Requirements for the currently runnable utilities

- Python 3.10 or newer
- NumPy

Create a small environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy
```

PyTorch is additionally required by `dataset.py`. A complete
`requirements.txt` will be added after the public execution path is complete.

## Generate an analytic billet point cloud

The uploaded sampler allocates the requested total number of points between
the cylindrical wall and the two caps in proportion to surface area. The two
caps always receive the same number of points.

```python
import numpy as np

from sampling import sample_initial_billet

X0, labels = sample_initial_billet(
    R0=1.0,
    H0=1.35,
    N=1020,
    rotate=False,
    seed=7,
    return_labels=True,
)

print(X0.shape)  # (1020, 3)
print(dict(zip(*np.unique(labels, return_counts=True))))
```

The wall is sampled uniformly in angle and axial coordinate. Each cap is
sampled uniformly by area using `r = R0 * sqrt(u)`.

## Convert raw SQLite data into shards

The input database must contain a `strike` table with at least these fields:

- `series_id`
- `result`, containing JSON arrays named `Steps` and `Vertices`
- `position`
- `rotation`, stored as a quaternion in `[x, y, z, w]` order

To extract every usable transition from every trajectory:

```bash
python sqlite_to_forgenet_shards.py \
  "/path/to/noisy_cogging.db" \
  --output-dir data/jax_fem_shards_all_v1 \
  --max-transitions 0 \
  --line-limit 0 \
  --points-per-state 1020 \
  --samples-per-shard 128 \
  --order-by-series
```

Both zero-valued limits are required for a complete database scan:

- `--max-transitions 0` removes the transition-count limit.
- `--line-limit 0` removes the row-scan limit.

Use a new or empty output directory. The extractor does not remove stale shard
files from an earlier run.

### How one transition is constructed

For two consecutive valid strike rows in the same trajectory:

1. Keep the final `Vertices` state from each row.
2. Sum the destination row's internal `Steps` to obtain scalar compression.
3. By default, apply the destination strike's recorded rotation and position
   to both endpoint states.
4. Compute `delta = X_next - X_t` after that alignment.
5. Store the transition and its trajectory metadata in an NPZ shard.

The first row of every trajectory is skipped because it has no preceding
state. Multiple internal solver steps inside one SQLite row are collapsed into
one endpoint transition; they do not become separate training samples.

The recorded-pose operation is deterministic preprocessing, not an ICP search
for the best alignment. Pass `--no-apply-pose` only when raw, unaligned
coordinates are intentionally required.

## Processed shard contract

Each shard contains `S` transition samples:

| Field | Shape | Meaning |
|---|---:|---|
| `X_t` | `(S, N, 3)` | Aligned point cloud before one transition. |
| `compression` | `(S, 1)` | Sum of the destination strike's internal compression steps. |
| `delta` | `(S, N, 3)` | Per-point displacement, `X_next - X_t`. |
| `X_next` | `(S, N, 3)` | Aligned point cloud after the transition. |
| `trajectory_id` | `(S,)` | Numeric trajectory group used for leakage-free splitting. |
| `trajectory_key` | `(S,)` | Original persistent series identifier. |
| `step_id` | `(S,)` | Source-row ordering metadata. |
| `position` | `(S, 3)` | Recorded strike-position metadata. |
| `rotation` | `(S, 4)` | Recorded strike quaternion metadata. |

`dataset.py` reconstructs `X_next` from `X_t + delta` and scales only the
displacement target by `delta_scalar` (100 by default). Stored `position` and
`rotation` remain metadata and are not passed to ForgeNet.

Training, validation, and test partitions must be created with
`make_trajectory_train_val_test_datasets`. Splitting individual transitions
would allow neighboring states from the same physical trajectory to leak
between partitions.

## Data availability

Raw SQLite databases, processed shards, model checkpoints, and generated
outputs are intentionally not stored in GitHub. A separate documented data
download will be added later. Do not commit local database paths or private
OneDrive locations.

## Known integration gaps

- The current public `dataset.py` imports `physics.py` and `transforms.py`,
  which have not yet been uploaded. Therefore the dataset module is not yet a
  standalone runnable public entry point.
- The synthetic demo helper at the bottom of `dataset.py` still calls the older
  `N_side`/`N_caps` sampler interface, while the uploaded `sampling.py` uses a
  single total `N`. The high-fidelity shard loader is unaffected, but the demo
  helper must be reconciled before release.
- Automated tests, dependency locking, ForgeNet, training, evaluation, and MPC
  are pending publication.

## Planned next steps

1. Reconcile the sampler and synthetic demo interfaces.
2. Publish the dataset tests and complete dependency specification.
3. Review and publish `model.py`.
4. Add a supported trajectory-safe ForgeNet training entry point.
5. Add evaluation and fixed-pose compression MPC only after their interfaces
   are verified.

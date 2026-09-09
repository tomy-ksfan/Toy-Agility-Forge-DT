# Toy Agility Forge Digital Twin

A work-in-progress point-cloud workflow for billet forging: prepare
high-fidelity JAX-FEM transition data, load it for learning, and predict
one-step deformation with ForgeNet.

## Published files

| File | Responsibility |
|---|---|
| [sampling.py](sampling.py) | Generate a cylindrical surface point cloud from radius, half-height, and total point count `N`. |
| [sqlite_to_forgenet_shards.py](sqlite_to_forgenet_shards.py) | Convert consecutive SQLite strike endpoints into NPZ transition shards. |
| [dataset.py](dataset.py) | Load shards lazily, construct training tensors, and split complete trajectories. |
| [model.py](model.py) | Define ForgeNet and helpers for scaled and physical displacement predictions. |
| [train_forgenet.py](train_forgenet.py) | Define initialization, batching, training, validation-based checkpoint selection, and final test evaluation. |
| [tests/test_dataset.py](tests/test_dataset.py) | Test shard loading, tensor construction, pose-metadata isolation, and trajectory-safe splitting. |

The data path is SQLite → NPZ shards → `dataset.py` → ForgeNet training.
The loader and trainer use the same trajectory-safe split interface. See
[Training](#training) for data inspection and training commands.

`sampling.py` is a separate analytic billet utility. It does not create
high-fidelity displacement labels or replace the FEM training data.

## Setup

The published Python files require Python 3.10 or newer, NumPy, and PyTorch.
The sampler and SQLite extractor need only NumPy beyond the standard library.

```bash
git clone https://github.com/tomy-ksfan/Toy-Agility-Forge-DT.git
cd Toy-Agility-Forge-DT
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy torch
```

The activation command above is for macOS/Linux. Dependencies are not yet
pinned in a requirements or lock file.

## Generate an initial billet

`R0` is the cylinder radius and `H0` is its half-height: the caps are at
`z=+H0` and `z=-H0`. The sampler accepts one required total point count,
`N`, and allocates it approximately in proportion to surface area:

- Each cap: `round(N * R0 / (2 * (2 * H0 + R0)))` points.
- Side wall: the remaining points after both caps.
- Wall sampling: uniform angle and axial coordinate.
- Cap sampling: uniform disk area, using `r = R0 * sqrt(u)`.

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

print(X0.shape)  # (1020, 3), float32
print(dict(zip(*np.unique(labels, return_counts=True))))
# side: 744, top_cap: 138, bottom_cap: 138
```

To match an already loaded target array `target_full` with shape `(M, 3)`,
derive the count in the caller:

```python
X0 = sample_initial_billet(
    R0=1.0,
    H0=1.35,
    N=target_full.shape[0],
    rotate=False,
    seed=7,
)
```

For an 8,108-point target and these dimensions, the billet has 5,916 side
points and 1,096 points per cap. The sampler does not accept `target_points`
or separate `N_side`/`N_caps` arguments.

Matching counts does not copy the target's shape, align the two objects, or
establish material-point correspondence. It also does not resample existing
FEM shards. `rotate=True` optionally rotates the sampled pattern around the
z axis; it is not a die-pose alignment search.

## Data availability

The processed high-fidelity JAX-FEM dataset is hosted separately from GitHub
because of its size:

- [JAX-FEM dataset on OSU OneDrive/SharePoint](https://buckeyemailosu-my.sharepoint.com/:f:/r/personal/fan_1317_osu_edu/Documents/JAX-FEM%20dataset?d=w75a47bf7a8234696a956b6dda7d5ee0e&csf=1&web=1&e=JzdxaV)

The reviewed all-trajectory dataset contains:

- 132,300 one-step transitions
- 1,847 forging trajectories
- 1,034 NPZ shards
- 1,020 aligned surface points per state
- Approximately 3.8 GB
- Offline pose alignment enabled (`apply_pose=true`)
- No target-based trajectory filtering

After downloading, preserve the directory structure:

```text
data/jax_fem_shards_all_v1/
├── extraction_metadata.json
├── README.md
├── shard_00000.npz
├── shard_00001.npz
└── ...
```

The raw SQLite database, trained checkpoints, and generated outputs are not
included in the repository. Access to the linked folder depends on its
SharePoint permissions.

## Convert raw SQLite data into shards

Skip this step if you already downloaded the processed shards. The input
database must contain a `strike` table with `series_id`, `result`,
`position`, and `rotation`. The `result` JSON contains `Steps` and
`Vertices`; rotations use quaternion order `[x, y, z, w]`.

To extract every usable transition without trajectory filtering:

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

Both zero-valued limits are needed: `--max-transitions 0` removes the
per-database transition limit, and `--line-limit 0` removes the row-scan
limit. Omit `--series-ids-file` to include all trajectories. Use a new or
empty output directory; the extractor does not remove stale shards.

For consecutive strike rows in the same trajectory, the extractor:

1. Keeps the final vertex state from each row.
2. Sums the destination row's internal `Steps` to obtain scalar compression.
3. Applies the destination strike's recorded rotation and position to both
   endpoints by default.
4. Uses the same retained vertex indices for both endpoints and computes
   `delta = X_next - X_t`.
5. Writes the transition and its metadata to a shard.

The first row of each trajectory supplies an initial endpoint, not a training
transition. Internal solver steps within a row are collapsed into its final
state; they do not become separate samples.

Vertex subsampling is random, with retained indices reused within each
trajectory; it is not FPS or triangle-area sampling. `--points-per-state 0`
keeps all vertices, but transitions stacked in one shard must have compatible
point counts. The documented 1,020-point setting matches the reviewed dataset.

Pose processing is deterministic offline preprocessing, not optimization for
the best alignment. `--no-apply-pose` disables it and should only be used
when unaligned coordinates are intentionally required.

## Shards and training tensors

With default pose processing, a shard containing `S` transitions has:

| Field | Shape | Meaning |
|---|---|---|
| `X_t` | `(S, N, 3)` | Aligned pre-transition coordinates. |
| `compression` | `(S, 1)` | Scalar compression for each transition. |
| `delta` | `(S, N, 3)` | Physical displacement, `X_next - X_t`. |
| `X_next` | `(S, N, 3)` | Aligned post-transition coordinates. |
| `trajectory_id` | `(S,)` | Numeric trajectory group. |
| `trajectory_key` | `(S,)` | Original series identifier. |
| `step_id` | `(S,)` | Source-row ordering metadata. |
| `source_db` | `(S,)` | Source database filename. |
| `position` | `(S, 3)` | Recorded strike-position metadata. |
| `rotation` | `(S, 4)` | Recorded strike-quaternion metadata. |

`dataset.py` reconstructs `X_next = X_t + delta` and returns four
float32 tensors per sample:

| Tensor | Per-sample shape | Meaning |
|---|---|---|
| State | `(N, 3)` | `X_t` |
| Action | `(1, 1)` | One scalar compression |
| Scaled displacement | `(1, N, 3)` | `delta * delta_scalar` |
| Next state | `(1, N, 3)` | `X_t + delta` |

`delta_scalar` defaults to 100. The leading length-one dimension is the
single transition dimension, not a full action sequence. A DataLoader adds
the batch dimension in front.

`position` and `rotation` remain metadata: they are not neural-network
inputs. The loader performs no additional runtime pose transformation and
rejects legacy datasets containing `theta` or `shift`.

```python
from dataset import open_training_dataset

dataset = open_training_dataset(
    "data/jax_fem_shards_all_v1",
    delta_scalar=100.0,
    shard_cache_size=2,
)
state, action, scaled_delta, next_state = dataset[0]
print(state.shape, action.shape, scaled_delta.shape, next_state.shape)
```

### Trajectory-safe split

Use `make_trajectory_train_val_test_datasets` to keep each complete trajectory
within one partition. The default split is approximately 80% training, 10%
validation, and 10% test **by trajectory count**, not by transition count.

- Validation and test counts are rounded from the total trajectory count,
  with at least one trajectory each; all remaining groups go to training.
- The longest trajectory occupies one of the test slots, not an extra slot.
  Ties are resolved by the smallest trajectory ID.
- Remaining assignments use a seeded shuffle. No trajectory is shared across
  partitions, and every transition is assigned exactly once.
- Optional `test_trajectory_id` forces another trajectory into the existing
  test allocation. It does not replace the longest or enlarge the test set;
  insufficient test slots produce an error.

```python
from dataset import make_trajectory_train_val_test_datasets

train_set, val_set, test_set, split_info = make_trajectory_train_val_test_datasets(
    dataset,
    val_fraction=0.1,
    test_fraction=0.1,
    seed=7,
)
assert split_info.longest_trajectory_id in split_info.test_trajectory_ids
```

For the reviewed corpus with the trainer defaults (`split_seed=7`):

| Partition | Trajectories | Transitions |
|---|---:|---:|
| Training | 1,477 | 106,423 |
| Validation | 185 | 12,951 |
| Test | 185 | 12,926 |
| Total | 1,847 | 132,300 |

The longest trajectory, ID 738, is included in test. Transition percentages
need not equal 80/10/10 because trajectory lengths differ. The split requires
at least three trajectories and enough groups to keep every partition nonempty.

## ForgeNet

`model.py` uses shared pointwise layers and global max pooling to encode
the cloud, an MLP to encode compression, and a pointwise decoder to predict
displacement. The trainer configures one action dimension.

- Direct `ForgeNet.forward`: cloud `(B, 3, N)`, action `(B, 1)`,
  scaled displacement output `(B, N, 3)`.
- `predict_scaled_delta`: accepts clouds as `(B, N, 3)`.
- `predict_physical_delta`: divides by `delta_scalar` and enforces zero
  displacement for a zero-compression hold action.

The helpers accept one-step actions shaped `(B,)`, `(B, 1)`, or
`(B, 1, 1)`. A multi-step tensor such as `(B, 5, 1)` is rejected rather
than silently using its first action.

A standalone shape check, using randomly initialized weights:

```python
import torch
from model import ForgeNet, predict_physical_delta

model = ForgeNet(
    point_size=0, latent_size=512, action_dims=1,
    dropout=0.0, use_res=False, delta_scalar=100.0,
).eval()
state = torch.zeros(1, 16, 3)
action = torch.tensor([[0.05]])

with torch.inference_mode():
    delta = predict_physical_delta(model, state, action)
    next_state = state + delta

print(next_state.shape)  # torch.Size([1, 16, 3])
```

This checks the interface, not prediction accuracy. A trained checkpoint is
required for meaningful deformation predictions.

## Training

`train_forgenet.py` loads a directory of offline-aligned NPZ shards and
uses the trajectory-safe splitter above. It starts from seeded random
initialization rather than loading an existing checkpoint.

Trainer defaults:

| Setting | Default |
|---|---|
| Data directory | `data/jax_fem_shards_all_v1` |
| Validation / test fractions | 0.1 / 0.1 of trajectory counts; longest trajectory included in test |
| Initialization / split seed | 17 / 7 |
| Batch size / shard cache | 64 / 2 shards |
| Training points | Up to 256 corresponding points per sample |
| Epochs / learning rate | 20 / 0.00015 |
| Optimizer | AdamW, weight decay `1e-6` |
| Device | `cpu` |

After downloading or extracting the shards, inspect the data first and
start training explicitly when ready:

```bash
# Inspect loading and split coverage without fitting or saving a model.
python train_forgenet.py --inspect-data --data data/jax_fem_shards_all_v1

# Train from seeded random initialization; the output directory must not exist.
python train_forgenet.py --train \
  --data data/jax_fem_shards_all_v1 \
  --output outputs/forgenet_run_001
```

Training shuffles shards and their training samples, visits every training
transition once per epoch, and merges a final singleton batch rather than
dropping it. It applies matching point indices to inputs and targets, uses
next-state point MSE with `delta_scalar ** 2` gradient scaling, clips
gradients, and schedules the learning rate with cosine annealing.

Validation and final test evaluation use all stored points. Checkpoint
selection uses validation MSE, including the initial epoch-zero baseline;
test data is evaluated only after selection. A completed run writes:

- `best.pt`: selected weights, configuration, split identities, history,
  and metrics.
- `metrics.json`: the run report and validation/test metrics.

The checkpoint contains inference weights, not optimizer state for resuming
training. This entry point does not perform pose alignment, recursive rollout,
or MPC.

## Available checks

Run the uploaded dataset tests without downloading the full dataset:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

All 13 published tests pass on temporary data. They cover shard loading,
tensor construction, metadata isolation, trajectory-count splitting,
longest-trajectory inclusion, seed reproducibility, and invalid split choices.

Before publication, nine additional local trainer-integration tests also
passed against the uploaded model/trainer and updated loader, including a
one-epoch synthetic run, checkpoint reload, and overwrite protection. These
additional tests are not included in the published test directory. The full
132,300-transition corpus passed a loading and split-coverage inspection.
These checks do not establish deformation accuracy or closed-loop performance;
full-corpus training was not run as part of this update.

The command-line help is also available without a dataset:

```bash
python sqlite_to_forgenet_shards.py --help
python train_forgenet.py --help
```

"""Load aligned high-fidelity displacement transitions for ForgeNet."""

from __future__ import annotations

import bisect
import warnings
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class HighFidelityDatasetArrays:
    """Container for aligned high-fidelity one-step transitions."""

    X_t: np.ndarray
    delta: np.ndarray
    compression: np.ndarray
    X_next: np.ndarray
    trajectory_id: np.ndarray | None = None
    step_id: np.ndarray | None = None


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    if hasattr(obj, "files"):
        return obj[name] if name in obj.files else None
    return getattr(obj, name, None)


def _to_numpy(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise KeyError(f"Required dataset field is missing: {name}")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_high_fidelity_dataset(path: str | Path) -> HighFidelityDatasetArrays:
    """Load a high-fidelity displacement dataset from ``.npz`` or ``.pt``.

    The required fields are ``X_t``, ``delta``, and ``compression``. The next
    state is always computed as ``X_next = X_t + delta`` so the displacement is
    never mistaken for an absolute coordinate field.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".npz":
        raw = np.load(path, allow_pickle=False)
    elif path.suffix in {".pt", ".pth"}:
        raw = torch.load(path, map_location="cpu")
    else:
        raise ValueError("Supported dataset formats are .npz, .pt, and .pth.")

    legacy_pose_fields = [
        name for name in ("theta", "shift") if _field(raw, name) is not None
    ]
    if legacy_pose_fields:
        raise ValueError(
            "Legacy runtime pose fields are not supported: "
            f"{legacy_pose_fields}. Align coordinates during offline extraction "
            "before loading them for training."
        )

    X_t = _to_numpy(_field(raw, "X_t"), "X_t").astype(np.float32)
    delta = _to_numpy(_field(raw, "delta"), "delta").astype(np.float32)
    compression = _to_numpy(_field(raw, "compression"), "compression").astype(np.float32)

    if X_t.ndim != 3 or X_t.shape[-1] != 3:
        raise ValueError(f"X_t must have shape (S, N, 3); received {X_t.shape}.")
    if delta.shape != X_t.shape:
        raise ValueError(f"delta must match X_t shape {X_t.shape}; received {delta.shape}.")

    if compression.ndim == 1:
        compression = compression[:, None]
    if compression.shape != (X_t.shape[0], 1):
        raise ValueError(
            f"compression must have shape (S, 1); received {compression.shape}."
        )

    X_next_from_delta = X_t + delta
    provided_X_next = _field(raw, "X_next")
    if provided_X_next is not None:
        provided = _to_numpy(provided_X_next, "X_next").astype(np.float32)
        max_mismatch = float(np.max(np.abs(provided - X_next_from_delta)))
        if max_mismatch > 1.0e-4:
            warnings.warn(
                "Provided X_next differs from X_t + delta; using X_t + delta as required. "
                f"max mismatch={max_mismatch:.3e}",
                RuntimeWarning,
            )

    trajectory_id = _field(raw, "trajectory_id")
    step_id = _field(raw, "step_id")

    return HighFidelityDatasetArrays(
        X_t=X_t,
        delta=delta,
        compression=compression,
        X_next=X_next_from_delta.astype(np.float32),
        trajectory_id=None if trajectory_id is None else np.asarray(trajectory_id),
        step_id=None if step_id is None else np.asarray(step_id),
    )


def prepare_training_arrays(arrays: HighFidelityDatasetArrays) -> dict[str, np.ndarray]:
    """Return aligned model inputs and targets without applying another pose.

    Coordinate-frame alignment is an offline extraction responsibility. This
    loader therefore treats ``X_t`` and ``delta`` as final training tensors;
    recorded pose fields remain metadata and never enter the network sample.
    """

    return {
        "X_t": arrays.X_t.astype(np.float32),
        "compression": arrays.compression.astype(np.float32),
        "delta": arrays.delta.astype(np.float32),
        "X_next": arrays.X_next.astype(np.float32),
    }


class PointCloudTransitionDataset(Dataset):
    """Repo-style transition dataset returning scaled deltas.

    Each item is ``(x_t, a.unsqueeze(0), delta_t.unsqueeze(0),
    x_tp1.unsqueeze(0))``, matching the OSU forge-net dataloader convention.
    ``delta_t`` is ``delta_scalar * (x_tp1 - x_t)``.
    """

    def __init__(self, arrays: dict[str, np.ndarray], delta_scalar: float = 100.0) -> None:
        self.X_t = torch.as_tensor(arrays["X_t"], dtype=torch.float32)
        self.compression = torch.as_tensor(arrays["compression"], dtype=torch.float32)
        self.delta = torch.as_tensor(arrays["delta"], dtype=torch.float32) * float(delta_scalar)
        self.X_next = torch.as_tensor(arrays["X_next"], dtype=torch.float32)
        self.delta_scalar = float(delta_scalar)

    def __len__(self) -> int:
        return int(self.X_t.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.X_t[idx],
            self.compression[idx].unsqueeze(0),
            self.delta[idx].unsqueeze(0),
            self.X_next[idx].unsqueeze(0),
        )


@dataclass(frozen=True)
class DatasetFieldInfo:
    """Shape and dtype metadata for one stored array."""

    shape: tuple[int, ...]
    dtype: np.dtype


@dataclass(frozen=True)
class ShardInfo:
    """Metadata for one sharded high-fidelity dataset file."""

    path: Path
    start: int
    stop: int
    fields: dict[str, DatasetFieldInfo]


@dataclass(frozen=True)
class TrajectorySplitInfo:
    """Trajectory identities and sample counts for a leakage-free split."""

    train_trajectory_ids: tuple[int, ...]
    validation_trajectory_ids: tuple[int, ...]
    test_trajectory_ids: tuple[int, ...]
    train_samples: int
    validation_samples: int
    test_samples: int


def _as_compression_array(compression: Any) -> np.ndarray:
    value = np.asarray(compression, dtype=np.float32)
    if value.ndim == 0:
        value = value.reshape(1)
    if value.ndim == 1 and value.shape[0] == 1:
        return value.astype(np.float32)
    if value.shape == (1, 1):
        return value.reshape(1).astype(np.float32)
    raise ValueError(f"compression sample must be scalar or shape (1,); received {value.shape}.")


def _prepare_one_training_sample(
    X_t: np.ndarray,
    delta: np.ndarray,
    compression: np.ndarray,
    delta_scalar: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare one aligned sample without applying a runtime pose transform."""

    X_t = np.asarray(X_t, dtype=np.float32)
    delta = np.asarray(delta, dtype=np.float32)
    compression = _as_compression_array(compression)
    if X_t.ndim != 2 or X_t.shape[-1] != 3:
        raise ValueError(f"X_t sample must have shape (N, 3); received {X_t.shape}.")
    if delta.shape != X_t.shape:
        raise ValueError(f"delta sample must match X_t shape {X_t.shape}; received {delta.shape}.")

    X_next = X_t + delta

    return (
        torch.as_tensor(X_t, dtype=torch.float32),
        torch.as_tensor(compression, dtype=torch.float32).unsqueeze(0),
        torch.as_tensor(delta * float(delta_scalar), dtype=torch.float32).unsqueeze(0),
        torch.as_tensor(X_next, dtype=torch.float32).unsqueeze(0),
    )


def _validate_required_field_shapes(fields: dict[str, DatasetFieldInfo], source: Path) -> int:
    legacy_pose_fields = [name for name in ("theta", "shift") if name in fields]
    if legacy_pose_fields:
        raise ValueError(
            f"{source}: legacy runtime pose fields are not supported: "
            f"{legacy_pose_fields}. Align coordinates during offline extraction."
        )

    missing = [name for name in ("X_t", "delta", "compression") if name not in fields]
    if missing:
        raise KeyError(f"{source} is missing required fields: {missing}")

    X_shape = fields["X_t"].shape
    delta_shape = fields["delta"].shape
    compression_shape = fields["compression"].shape
    if len(X_shape) != 3 or X_shape[-1] != 3:
        raise ValueError(f"{source}: X_t must have shape (S, N, 3); received {X_shape}.")
    if delta_shape != X_shape:
        raise ValueError(f"{source}: delta must match X_t shape {X_shape}; received {delta_shape}.")
    if len(compression_shape) == 1:
        if compression_shape[0] != X_shape[0]:
            raise ValueError(f"{source}: compression length must match X_t samples.")
    elif compression_shape != (X_shape[0], 1):
        raise ValueError(
            f"{source}: compression must have shape (S,) or (S, 1); received {compression_shape}."
        )
    if "position" in fields and fields["position"].shape != (X_shape[0], 3):
        raise ValueError(f"{source}: position must have shape (S, 3).")
    if "rotation" in fields and fields["rotation"].shape != (X_shape[0], 4):
        raise ValueError(f"{source}: rotation must have shape (S, 4).")
    if "trajectory_id" in fields and fields["trajectory_id"].shape != (X_shape[0],):
        raise ValueError(f"{source}: trajectory_id must have shape (S,).")
    return X_shape[0]


def _npy_field_info(path: Path) -> DatasetFieldInfo:
    arr = np.load(path, mmap_mode="r")
    return DatasetFieldInfo(shape=tuple(arr.shape), dtype=arr.dtype)


def _npz_field_info(path: Path) -> dict[str, DatasetFieldInfo]:
    """Read .npz array shapes and dtypes without loading array payloads."""

    fields: dict[str, DatasetFieldInfo] = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.endswith(".npy"):
                continue
            field = Path(name).stem
            with zf.open(name) as f:
                version = np.lib.format.read_magic(f)
                shape, _, dtype = np.lib.format._read_array_header(f, version)
            fields[field] = DatasetFieldInfo(shape=tuple(shape), dtype=np.dtype(dtype))
    return fields


class _MemmapArrayStore:
    """Lazy store for a directory containing X_t.npy, delta.npy, compression.npy."""

    optional_fields = (
        "X_next",
        "trajectory_id",
        "step_id",
        "position",
        "rotation",
    )

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.arrays: dict[str, np.ndarray] = {}
        for field in ("X_t", "delta", "compression", *self.optional_fields):
            path = self.root / f"{field}.npy"
            if path.exists():
                self.arrays[field] = np.load(path, mmap_mode="r")

        fields = {
            name: DatasetFieldInfo(shape=tuple(array.shape), dtype=array.dtype)
            for name, array in self.arrays.items()
        }
        self.length = _validate_required_field_shapes(fields, self.root)
        self.fields = fields

    def get(self, idx: int) -> dict[str, np.ndarray]:
        return {name: array[idx] for name, array in self.arrays.items()}

    def read_field(self, name: str) -> np.ndarray:
        if name not in self.arrays:
            raise KeyError(f"Dataset metadata field is missing: {name}")
        return np.asarray(self.arrays[name])


class _NpzShardStore:
    """Lazy store for a directory of smaller .npz shards."""

    def __init__(self, root: Path, cache_size: int = 2) -> None:
        self.root = Path(root)
        self.cache_size = max(1, int(cache_size))
        shard_paths = sorted(self.root.glob("*.npz"))
        if not shard_paths:
            raise FileNotFoundError(f"No .npz shards found in {self.root}.")

        self.shards: list[ShardInfo] = []
        start = 0
        for path in shard_paths:
            fields = _npz_field_info(path)
            count = _validate_required_field_shapes(fields, path)
            self.shards.append(ShardInfo(path=path, start=start, stop=start + count, fields=fields))
            start += count
        self.length = start
        self._stops = [shard.stop for shard in self.shards]
        self._cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

    def _load_shard(self, path: Path) -> dict[str, np.ndarray]:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]

        with np.load(path, allow_pickle=False) as raw:
            arrays = {
                name: raw[name].astype(np.float32, copy=False)
                for name in ("X_t", "delta", "compression")
                if name in raw.files
            }
        self._cache[path] = arrays
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return arrays

    def get(self, idx: int) -> dict[str, np.ndarray]:
        shard_idx = bisect.bisect_right(self._stops, idx)
        shard = self.shards[shard_idx]
        local_idx = idx - shard.start
        arrays = self._load_shard(shard.path)
        return {name: array[local_idx] for name, array in arrays.items()}

    def read_field(self, name: str) -> np.ndarray:
        """Read one small metadata field across shards without loading point clouds."""

        values: list[np.ndarray] = []
        for shard in self.shards:
            if name not in shard.fields:
                raise KeyError(f"{shard.path} is missing dataset metadata field: {name}")
            with np.load(shard.path, allow_pickle=False) as raw:
                values.append(np.asarray(raw[name]))
        return np.concatenate(values, axis=0)


class LazyHighFidelityTransitionDataset(Dataset):
    """Streaming dataset for real high-fidelity JAX-FEM-scale data.

    Supported layouts
    -----------------
    1. Memory-mapped array directory::

           root/
             X_t.npy
             delta.npy
             compression.npy

       Arrays are opened with ``mmap_mode=\"r\"`` and sampled lazily.

    2. Sharded ``.npz`` directory::

           root/
             shard_00000.npz
             shard_00001.npz
             ...

       Each shard must contain ``X_t``, ``delta``, and ``compression``. Keep
       shards moderately sized because a shard is decompressed into an LRU cache.
    """

    def __init__(
        self,
        path: str | Path,
        delta_scalar: float = 100.0,
        max_samples: int | None = None,
        shard_cache_size: int = 2,
    ) -> None:
        self.path = Path(path)
        self.delta_scalar = float(delta_scalar)

        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if self.path.is_dir() and (self.path / "X_t.npy").exists():
            self.store = _MemmapArrayStore(self.path)
            self.layout = "memmap_npy"
        elif self.path.is_dir():
            self.store = _NpzShardStore(self.path, cache_size=shard_cache_size)
            self.layout = "npz_shards"
        else:
            raise ValueError(
                "LazyHighFidelityTransitionDataset expects a directory of .npy arrays "
                "or a directory of .npz shards."
            )

        self._length = self.store.length
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive when provided.")
            self._length = min(self._length, int(max_samples))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)
        fields = self.store.get(idx)
        return _prepare_one_training_sample(
            X_t=fields["X_t"],
            delta=fields["delta"],
            compression=fields["compression"],
            delta_scalar=self.delta_scalar,
        )

    def metadata(self, name: str) -> np.ndarray:
        """Return a sample-aligned metadata field, respecting ``max_samples``."""

        return np.asarray(self.store.read_field(name))[: self._length]


class StridedSplitDataset(Dataset):
    """Memory-light train/validation split wrapper.

    Validation samples are every ``period``-th base sample starting at index 0.
    Training samples are all remaining indices. This avoids storing a shuffled
    list of millions of indices for very large high-fidelity datasets.
    """

    def __init__(self, base: Dataset, split: str, period: int = 5) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'.")
        if period < 2:
            raise ValueError("period must be at least 2.")
        self.base = base
        self.split = split
        self.period = period
        n = len(base)
        self.val_count = (n + period - 1) // period
        self.train_count = n - self.val_count

    def __len__(self) -> int:
        return self.val_count if self.split == "val" else self.train_count

    def _global_index(self, idx: int) -> int:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        if self.split == "val":
            return idx * self.period
        block = idx // (self.period - 1)
        offset = idx % (self.period - 1) + 1
        return block * self.period + offset

    def __getitem__(self, idx: int) -> Any:
        return self.base[self._global_index(idx)]


class IndexedTransitionDataset(Dataset):
    """Dataset view backed by explicit global indices."""

    def __init__(self, base: Dataset, indices: np.ndarray) -> None:
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, idx: int) -> Any:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        return self.base[int(self.indices[idx])]


def make_train_val_datasets(
    dataset: Dataset,
    val_fraction: float = 0.2,
) -> tuple[Dataset, Dataset]:
    """Create train/validation datasets without materializing split indices."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    period = max(2, int(round(1.0 / val_fraction)))
    return (
        StridedSplitDataset(dataset, split="train", period=period),
        StridedSplitDataset(dataset, split="val", period=period),
    )


def make_trajectory_train_val_test_datasets(
    dataset: Dataset,
    val_fraction: float = 0.2,
    seed: int = 0,
    test_trajectory_id: int | None = None,
) -> tuple[Dataset, Dataset, Dataset, TrajectorySplitInfo]:
    """Split complete trajectories so neighboring states cannot leak across sets.

    When no test trajectory is specified, the longest trajectory is held out so
    the test set supports the longest possible recursive evaluation. Validation
    trajectories are then selected deterministically from the remaining groups.
    """

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    metadata = getattr(dataset, "metadata", None)
    if metadata is None:
        raise TypeError(
            "Trajectory splitting requires a streaming dataset with trajectory_id metadata."
        )

    trajectory_ids = np.asarray(metadata("trajectory_id"), dtype=np.int64).reshape(-1)
    if trajectory_ids.shape[0] != len(dataset):
        raise ValueError("trajectory_id metadata length does not match the dataset.")
    unique_ids, counts = np.unique(trajectory_ids, return_counts=True)
    if unique_ids.size < 3:
        raise ValueError("At least three trajectories are required for train/validation/test splits.")

    count_by_id = {int(key): int(value) for key, value in zip(unique_ids, counts)}
    if test_trajectory_id is None:
        test_trajectory_id = int(unique_ids[int(np.argmax(counts))])
    if int(test_trajectory_id) not in count_by_id:
        raise ValueError(f"Unknown test_trajectory_id={test_trajectory_id}.")

    candidates = [int(value) for value in unique_ids if int(value) != int(test_trajectory_id)]
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    remaining_samples = len(dataset) - count_by_id[int(test_trajectory_id)]
    target_validation_samples = max(1, int(round(val_fraction * remaining_samples)))
    validation_ids: list[int] = []
    validation_count = 0
    remaining_candidates = candidates.copy()
    while remaining_candidates:
        current_gap = abs(target_validation_samples - validation_count)
        candidate = min(
            remaining_candidates,
            key=lambda value: abs(
                target_validation_samples - (validation_count + count_by_id[value])
            ),
        )
        candidate_gap = abs(
            target_validation_samples - (validation_count + count_by_id[candidate])
        )
        if validation_ids and candidate_gap >= current_gap:
            break
        validation_ids.append(candidate)
        validation_count += count_by_id[candidate]
        remaining_candidates.remove(candidate)
        if len(remaining_candidates) <= 1:
            break

    validation_set = set(validation_ids)
    test_set = {int(test_trajectory_id)}
    train_ids = [
        int(value)
        for value in unique_ids
        if int(value) not in validation_set and int(value) not in test_set
    ]
    if not train_ids or not validation_ids:
        raise ValueError("Trajectory split produced an empty train or validation set.")

    train_indices = np.flatnonzero(np.isin(trajectory_ids, train_ids))
    validation_indices = np.flatnonzero(np.isin(trajectory_ids, validation_ids))
    test_indices = np.flatnonzero(trajectory_ids == int(test_trajectory_id))
    info = TrajectorySplitInfo(
        train_trajectory_ids=tuple(sorted(train_ids)),
        validation_trajectory_ids=tuple(sorted(validation_ids)),
        test_trajectory_ids=(int(test_trajectory_id),),
        train_samples=int(train_indices.size),
        validation_samples=int(validation_indices.size),
        test_samples=int(test_indices.size),
    )
    return (
        IndexedTransitionDataset(dataset, train_indices),
        IndexedTransitionDataset(dataset, validation_indices),
        IndexedTransitionDataset(dataset, test_indices),
        info,
    )


def open_training_dataset(
    path: str | Path,
    delta_scalar: float = 100.0,
    max_samples: int | None = None,
    shard_cache_size: int = 2,
    eager_file_size_limit_gb: float = 4.0,
) -> Dataset:
    """Open a training dataset without assuming it fits in memory.

    Small single-file ``.npz``/``.pt`` datasets still use the eager loader for
    convenience. Directories use the streaming dataset for large real data.
    """

    path = Path(path)
    if path.is_dir():
        return LazyHighFidelityTransitionDataset(
            path=path,
            delta_scalar=delta_scalar,
            max_samples=max_samples,
            shard_cache_size=shard_cache_size,
        )

    if eager_file_size_limit_gb > 0:
        size_gb = path.stat().st_size / (1024**3)
        if size_gb > eager_file_size_limit_gb:
            raise ValueError(
                f"{path} is {size_gb:.1f} GB. Single-file datasets are loaded eagerly; "
                "use a directory of memory-mapped .npy arrays or .npz shards for large data."
            )

    arrays = load_high_fidelity_dataset(path)
    training_arrays = prepare_training_arrays(arrays)
    if max_samples is not None:
        training_arrays = {
            name: values[:max_samples]
            for name, values in training_arrays.items()
        }
    return PointCloudTransitionDataset(training_arrays, delta_scalar=delta_scalar)


def write_npz_shards_from_arrays(
    output_dir: str | Path,
    arrays: HighFidelityDatasetArrays,
    samples_per_shard: int,
) -> list[Path]:
    """Write an in-memory dataset into smaller .npz shards.

    This is intended for moderate conversion jobs and tests. For a 100+ GB
    export, prefer writing shards directly from the JAX-FEM generation script so
    the full dataset is never materialized in Python.
    """

    if samples_per_shard <= 0:
        raise ValueError("samples_per_shard must be positive.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    total = arrays.X_t.shape[0]
    for shard_idx, start in enumerate(range(0, total, samples_per_shard)):
        stop = min(start + samples_per_shard, total)
        payload: dict[str, np.ndarray] = {
            "X_t": arrays.X_t[start:stop],
            "delta": arrays.delta[start:stop],
            "compression": arrays.compression[start:stop],
            "X_next": arrays.X_next[start:stop],
        }
        for name in ("trajectory_id", "step_id"):
            value = getattr(arrays, name)
            if value is not None:
                payload[name] = value[start:stop]
        path = output / f"shard_{shard_idx:05d}.npz"
        np.savez_compressed(path, **payload)
        paths.append(path)
    return paths

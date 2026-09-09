"""Train the compression-only ForgeNet on offline-aligned transition shards.

Use --inspect-data for a read-only loading check, or --train explicitly to fit
a new model. This entry point does not run alignment, recursive rollout or MPC,
and never automatically loads the historical targeted checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from dataset import (
    IndexedTransitionDataset,
    LazyHighFidelityTransitionDataset,
    TrajectorySplitInfo,
    make_trajectory_train_val_test_datasets,
    open_training_dataset,
)
from model import ForgeNet, predict_physical_delta, predict_scaled_delta


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrainingConfig:
    data_path: str = str(ROOT / "data/jax_fem_shards_all_v1")
    batch_size: int = 64
    shard_cache_size: int = 2
    seed: int = 17
    split_seed: int = 7
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    # Optional extra test ID; the longest trajectory is always included.
    test_trajectory_id: int | None = None
    delta_scalar: float = 100.0
    training_points: int = 256
    epochs: int = 20
    learning_rate: float = 1.5e-4
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2 for ForgeNet BatchNorm.")
        if min(self.shard_cache_size, self.training_points, self.epochs) < 1:
            raise ValueError("shard_cache_size, training_points and epochs must be positive.")
        if not math.isfinite(self.delta_scalar) or self.delta_scalar <= 0:
            raise ValueError("delta_scalar must be finite and positive.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not 0 < self.val_fraction < 1:
            raise ValueError("val_fraction must be between 0 and 1.")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1.")
        if self.val_fraction + self.test_fraction >= 1:
            raise ValueError("val_fraction + test_fraction must be less than 1.")
        if self.seed < 0 or self.split_seed < 0:
            raise ValueError("Seeds must be nonnegative.")


class ShardGroupedSampler(Sampler[int]):
    """Shuffle shards and samples within each shard, using subset-local indices.

Every training transition is visited exactly once per epoch. Validation/test
indices are never added, even when their transitions share a shard with train.
"""

    def __init__(self, dataset: IndexedTransitionDataset, seed: int = 17) -> None:
        if getattr(dataset.base, "layout", None) != "npz_shards":
            raise ValueError("ShardGroupedSampler requires an NPZ-shard dataset view.")
        stops = np.asarray([shard.stop for shard in dataset.base.store.shards])
        shard_ids = np.searchsorted(stops, dataset.indices, side="right")
        # These are positions WITHIN the split, not global corpus indices.
        local_order = np.argsort(shard_ids, kind="stable")
        boundaries = np.flatnonzero(np.diff(shard_ids[local_order])) + 1
        self.groups = [group for group in np.split(local_order, boundaries) if group.size]
        self.length = len(dataset)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        for group_id in rng.permutation(len(self.groups)):
            yield from (int(index) for index in rng.permutation(self.groups[group_id]))


class NoSingletonBatchSampler(Sampler[list[int]]):
    """Keep all samples; merge a final singleton into the preceding batch.

The final batch may contain batch_size + 1 samples. This avoids the action
encoder's training-time BatchNorm error without dropping any transitions.
"""

    def __init__(self, sampler: Sampler[int], batch_size: int) -> None:
        if batch_size < 2 or len(sampler) < 2:
            raise ValueError("Training needs batch_size >= 2 and at least two samples.")
        self.sampler = sampler
        self.batch_size = batch_size

    def __len__(self) -> int:
        full, remainder = divmod(len(self.sampler), self.batch_size)
        return full + int(remainder > 1 or (remainder == 1 and full == 0))

    def __iter__(self) -> Iterator[list[int]]:
        pending: list[int] = []
        for index in self.sampler:
            pending.append(index)
            if len(pending) == self.batch_size + 2:
                yield pending[: self.batch_size]
                pending = pending[self.batch_size :]
        if pending:
            yield pending


@dataclass
class TrainingData:
    dataset: LazyHighFidelityTransitionDataset
    split: TrajectorySplitInfo
    sampler: ShardGroupedSampler
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader


def prepare_data(config: TrainingConfig) -> TrainingData:
    data_path = Path(config.data_path)
    if not data_path.is_dir():
        raise ValueError("This trainer requires a directory of offline-aligned NPZ shards.")
    dataset = open_training_dataset(
        data_path, delta_scalar=config.delta_scalar,
        shard_cache_size=config.shard_cache_size,
    )
    if getattr(dataset, "layout", None) != "npz_shards":
        raise ValueError("This trainer currently supports the NPZ-shard layout only.")
    train, validation, test, split = make_trajectory_train_val_test_datasets(
        dataset, val_fraction=config.val_fraction, seed=config.split_seed,
        test_trajectory_id=config.test_trajectory_id, test_fraction=config.test_fraction,
    )
    sampler = ShardGroupedSampler(train, seed=config.seed)
    # A single loading process keeps the shard cache and grouped access together.
    train_loader = DataLoader(
        train, batch_sampler=NoSingletonBatchSampler(sampler, config.batch_size),
        num_workers=0,
    )
    return TrainingData(
        dataset, split, sampler, train_loader,
        DataLoader(validation, batch_size=config.batch_size, shuffle=False, num_workers=0),
        DataLoader(test, batch_size=config.batch_size, shuffle=False, num_workers=0),
    )


def inspect_data(data: TrainingData) -> dict[str, object]:
    state, action, delta, next_state = next(iter(data.train_loader))
    if not all(torch.isfinite(value).all() for value in (state, action, delta, next_state)):
        raise ValueError("The inspected training batch contains non-finite values.")
    splits = [data.train_loader.dataset, data.validation_loader.dataset, data.test_loader.dataset]
    indices = np.concatenate([split.indices for split in splits])
    if not np.array_equal(np.sort(indices), np.arange(len(data.dataset))):
        raise ValueError("The split must cover the full corpus exactly once.")
    groups = [set(data.split.train_trajectory_ids), set(data.split.validation_trajectory_ids),
              set(data.split.test_trajectory_ids)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("Trajectory IDs must not overlap across splits.")
    if data.split.longest_trajectory_id not in groups[2]:
        raise ValueError("The longest trajectory must belong to test.")
    sampled = np.fromiter(data.sampler, dtype=np.int64)
    if not np.array_equal(np.sort(sampled), np.arange(len(splits[0]))):
        raise ValueError("The training sampler must visit each training sample exactly once.")
    return {
        "data_path": str(data.dataset.path.resolve()),
        "transitions": len(data.dataset),
        "shards": len(data.dataset.store.shards),
        "train_samples": len(splits[0]),
        "validation_samples": len(splits[1]),
        "test_samples": len(splits[2]),
        "trajectory_counts": [
            len(data.split.train_trajectory_ids), len(data.split.validation_trajectory_ids),
            len(data.split.test_trajectory_ids),
        ],
        "test_trajectory_ids": list(data.split.test_trajectory_ids),
        "longest_trajectory_id": data.split.longest_trajectory_id,
        "split_fraction_basis": "trajectory count, not transition count",
        "longest_trajectory_in_test": True,
        "training_shard_groups": len(data.sampler.groups),
        "training_batches": len(data.train_loader),
        "batch_shapes": {
            "X_t": list(state.shape), "compression": list(action.shape),
            "scaled_delta": list(delta.shape), "X_next": list(next_state.shape),
        },
        "batch_reconstruction_max_error": float(
            (next_state[:, 0] - state - delta[:, 0] / data.dataset.delta_scalar).abs().max()
        ),
        "coverage": "all corpus samples assigned; all training samples visited once per epoch",
        "runtime_pose_alignment": False,
    }


def model_config(delta_scalar: float) -> dict[str, object]:
    return dict(
        point_size=0, latent_size=512, action_dims=1,
        dropout=0.0, use_res=False, delta_scalar=delta_scalar,
    )


def initialize_model(config: TrainingConfig) -> ForgeNet:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    return ForgeNet(**model_config(config.delta_scalar)).to(config.device)


def evaluate(model: ForgeNet, loader: DataLoader) -> dict[str, float | None]:
    """Full-point physical-delta MSE with the existing zero-compression hold rule."""
    model.eval()
    device = next(model.parameters()).device
    squared_error = persistence_error = 0.0
    values = 0
    with torch.inference_mode():
        for state, action, delta_scaled, _ in loader:
            state, action = state.to(device), action.to(device)
            truth = delta_scaled[:, 0].to(device) / model.delta_scalar
            prediction = predict_physical_delta(model, state, action)
            squared_error += float(((prediction - truth) ** 2).sum())
            persistence_error += float((truth ** 2).sum())
            values += truth.numel()
    if not values or not math.isfinite(squared_error + persistence_error):
        raise ValueError("Evaluation requires nonempty, finite predictions and targets.")
    return {
        "mse": squared_error / values,
        "persistence_mse": persistence_error / values,
        "skill": 1 - squared_error / persistence_error if persistence_error > 0 else None,
    }


def train_one_epoch(
    model: ForgeNet, loader: DataLoader, optimizer: torch.optim.Optimizer,
    training_points: int,
) -> float:
    model.train()
    device = next(model.parameters()).device
    squared_error = 0.0
    values = 0
    for state, action, _, next_state in loader:
        state, action, next_state = state.to(device), action.to(device), next_state[:, 0].to(device)
        point_indices = torch.randperm(state.shape[1], device=device)[:training_points]
        # Apply exactly the same point indices to the input and its target.
        state, truth = state[:, point_indices], next_state[:, point_indices]
        optimizer.zero_grad(set_to_none=True)
        prediction = state + predict_scaled_delta(model, state, action) / model.delta_scalar
        point_mse = F.mse_loss(prediction, truth)
        if not torch.isfinite(point_mse):
            raise ValueError("Training produced a non-finite loss.")
        (point_mse * model.delta_scalar ** 2).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        squared_error += float(point_mse.detach()) * truth.numel()
        values += truth.numel()
    if not values:
        raise ValueError("Training loader is empty.")
    return squared_error / values


def train(config: TrainingConfig, data: TrainingData, output_dir: Path) -> Path:
    # A dedicated new run directory protects existing checkpoints from overwrite.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    model = initialize_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=min(1e-5, config.learning_rate),
    )
    baseline = evaluate(model, data.validation_loader)
    best_validation = baseline["mse"]
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        data.sampler.set_epoch(epoch - 1)
        mse = train_one_epoch(model, data.train_loader, optimizer, config.training_points)
        validation = evaluate(model, data.validation_loader)
        record = dict(epoch=epoch, train_subset_mse=mse, validation=validation,
                      learning_rate=optimizer.param_groups[0]["lr"])
        history.append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
        if validation["mse"] < best_validation:
            best_validation = validation["mse"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        scheduler.step()

    model.load_state_dict(best_state)
    # The test set is evaluated only after validation-based model selection.
    metrics = {"validation": evaluate(model, data.validation_loader),
               "test": evaluate(model, data.test_loader)}
    report = dict(
        config=asdict(config), model_config=model_config(config.delta_scalar),
        initialization={"mode": "random", "seed": config.seed, "source_checkpoint": None},
        split_info=asdict(data.split), best_epoch=best_epoch,
        split_policy="seeded trajectory-count split; longest forced into test",
        baseline_validation=baseline, history=history, metrics=metrics,
        objective="next-state point MSE multiplied by delta_scalar squared",
        optimizer_config={"name": "AdamW", "weight_decay": 1e-6, "gradient_clip_norm": 5.0},
        scheduler_config={"name": "CosineAnnealingLR", "T_max": config.epochs,
                          "eta_min": min(1e-5, config.learning_rate)},
        evaluation="full-point physical delta; zero-compression hold rule",
        checkpoint_kind="inference weights, not a resumable optimizer checkpoint",
    )
    checkpoint_path = output_dir / "best.pt"
    torch.save({**report, "model_state_dict": {
        name: tensor.detach().cpu() for name, tensor in best_state.items()
    }}, checkpoint_path)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return checkpoint_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect-data", action="store_true", help="Check data without training or saving files.")
    mode.add_argument("--train", action="store_true", help="Explicitly start training from random initialization.")
    parser.add_argument("--data", default=TrainingConfig.data_path)
    parser.add_argument("--output", type=Path, help="New run directory; required with --train.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-cache-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of all trajectories for validation (default: 0.1).")
    parser.add_argument("--test-fraction", type=float, default=0.1,
                        help="Fraction of all trajectories for test, including the longest (default: 0.1).")
    parser.add_argument("--test-trajectory-id", type=int,
                        help="Optional additional test trajectory; never replaces the longest.")
    parser.add_argument("--delta-scalar", type=float, default=100.0)
    parser.add_argument("--training-points", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.train and args.output is None:
        parser.error("--train requires --output pointing to a new run directory.")
    if args.inspect_data and args.output is not None:
        parser.error("--inspect-data does not create output files; omit --output.")
    config = TrainingConfig(**{
        "data_path": str(Path(args.data).resolve()),
        **{name: getattr(args, name) for name in TrainingConfig.__dataclass_fields__ if name != "data_path"},
    })
    data = prepare_data(config)
    print(json.dumps(inspect_data(data), indent=2, allow_nan=False), flush=True)
    if args.train:
        print(f"Saved checkpoint: {train(config, data, args.output)}", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    IndexedTransitionDataset,
    LazyHighFidelityTransitionDataset,
    make_trajectory_train_val_test_datasets,
    open_training_dataset,
)


def write_test_shard(
    root: Path,
    shard_number: int,
    sample_ids: list[int],
    trajectory_ids: list[int],
    *,
    point_count: int = 4,
) -> Path:
    """Write a small shard using the schema of the all-trajectory corpus."""

    if len(sample_ids) != len(trajectory_ids):
        raise ValueError("sample_ids and trajectory_ids must have equal lengths.")

    count = len(sample_ids)
    X_t = np.zeros((count, point_count, 3), dtype=np.float32)
    delta = np.zeros_like(X_t)
    compression = np.zeros((count, 1), dtype=np.float32)

    for local_index, sample_id in enumerate(sample_ids):
        X_t[local_index, :, 0] = float(sample_id)
        X_t[local_index, :, 1] = np.arange(point_count, dtype=np.float32)
        X_t[local_index, :, 2] = -float(sample_id)
        delta[local_index, :, 0] = 0.01 * float(sample_id + 1)
        compression[local_index, 0] = 0.1 * float(sample_id + 1)

    # Deliberately store an incorrect X_next. The training loader contract is
    # to reconstruct it from X_t + delta instead of trusting this field.
    stored_X_next = np.full_like(X_t, 999.0)
    position = np.column_stack(
        [
            np.asarray(sample_ids, dtype=np.float32) + 10.0,
            np.zeros(count, dtype=np.float32),
            np.zeros(count, dtype=np.float32),
        ]
    )
    rotation = np.tile(
        np.asarray([0.70710677, 0.0, 0.0, 0.70710677], dtype=np.float32),
        (count, 1),
    )

    path = root / f"shard_{shard_number:05d}.npz"
    np.savez_compressed(
        path,
        X_t=X_t,
        delta=delta,
        compression=compression,
        X_next=stored_X_next,
        trajectory_id=np.asarray(trajectory_ids, dtype=np.int64),
        trajectory_key=np.asarray(
            [f"trajectory-{value}" for value in trajectory_ids], dtype="U32"
        ),
        step_id=np.asarray(sample_ids, dtype=np.int64),
        source_db=np.full(count, "test.db", dtype="U16"),
        position=position,
        rotation=rotation,
    )
    return path


class LazyDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_crosses_shard_boundary_and_reads_partial_final_shard(self) -> None:
        write_test_shard(self.root, 0, [0, 1], [0, 0])
        write_test_shard(self.root, 1, [2], [1])

        dataset = open_training_dataset(self.root, delta_scalar=10.0)

        self.assertIsInstance(dataset, LazyHighFidelityTransitionDataset)
        self.assertEqual(dataset.layout, "npz_shards")
        self.assertEqual(len(dataset), 3)

        first = dataset[0]
        boundary = dataset[2]
        last = dataset[-1]
        for left, right in zip(boundary, last):
            torch.testing.assert_close(left, right)

        X_t, compression, delta_scaled, X_next = boundary
        self.assertEqual(tuple(X_t.shape), (4, 3))
        self.assertEqual(tuple(compression.shape), (1, 1))
        self.assertEqual(tuple(delta_scaled.shape), (1, 4, 3))
        self.assertEqual(tuple(X_next.shape), (1, 4, 3))
        torch.testing.assert_close(compression, torch.tensor([[0.3]]))
        torch.testing.assert_close(
            delta_scaled[0, :, 0], torch.full((4,), 0.3), rtol=1.0e-6, atol=1.0e-6
        )
        torch.testing.assert_close(X_next[0], X_t + delta_scaled[0] / 10.0)
        self.assertFalse(torch.any(X_next == 999.0))

        loader = DataLoader(dataset, batch_size=3, shuffle=False)
        batch_X, batch_action, batch_delta, batch_next = next(iter(loader))
        self.assertEqual(tuple(batch_X.shape), (3, 4, 3))
        self.assertEqual(tuple(batch_action.shape), (3, 1, 1))
        self.assertEqual(tuple(batch_delta.shape), (3, 1, 4, 3))
        self.assertEqual(tuple(batch_next.shape), (3, 1, 4, 3))
        torch.testing.assert_close(batch_X[0], first[0])

    def test_position_and_rotation_remain_metadata_only(self) -> None:
        write_test_shard(self.root, 0, [0, 1], [0, 0])

        dataset = open_training_dataset(self.root)
        X_t, _, _, X_next = dataset[1]

        expected_X_t = torch.tensor(
            [
                [1.0, 0.0, -1.0],
                [1.0, 1.0, -1.0],
                [1.0, 2.0, -1.0],
                [1.0, 3.0, -1.0],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(X_t, expected_X_t)
        torch.testing.assert_close(X_next[0, :, 0], torch.full((4,), 1.02))
        self.assertNotIn(
            "train_in_die_frame",
            inspect.signature(open_training_dataset).parameters,
        )

        np.testing.assert_allclose(
            dataset.metadata("position")[0], np.asarray([10.0, 0.0, 0.0])
        )
        np.testing.assert_allclose(
            dataset.metadata("rotation")[0],
            np.asarray([0.70710677, 0.0, 0.0, 0.70710677]),
        )

    def test_max_samples_limits_samples_and_metadata_to_the_same_prefix(self) -> None:
        write_test_shard(self.root, 0, [0, 1, 2], [0, 0, 1])
        write_test_shard(self.root, 1, [3, 4], [1, 2])

        dataset = open_training_dataset(self.root, max_samples=3)

        self.assertEqual(len(dataset), 3)
        np.testing.assert_array_equal(dataset.metadata("step_id"), [0, 1, 2])
        self.assertEqual(float(dataset[-1][0][0, 0]), 2.0)
        with self.assertRaises(IndexError):
            _ = dataset[3]

    def test_missing_required_shard_field_fails_during_open(self) -> None:
        np.savez_compressed(
            self.root / "shard_00000.npz",
            X_t=np.zeros((1, 4, 3), dtype=np.float32),
            delta=np.zeros((1, 4, 3), dtype=np.float32),
        )

        with self.assertRaisesRegex(KeyError, "compression"):
            open_training_dataset(self.root)

    def test_legacy_runtime_pose_fields_fail_for_shards_and_memmaps(self) -> None:
        X_t = np.zeros((1, 4, 3), dtype=np.float32)
        np.savez_compressed(
            self.root / "shard_00000.npz",
            X_t=X_t,
            delta=np.zeros_like(X_t),
            compression=np.zeros((1, 1), dtype=np.float32),
            theta=np.zeros((1, 1), dtype=np.float32),
            shift=np.zeros((1, 2), dtype=np.float32),
        )

        with self.assertRaisesRegex(ValueError, "offline extraction"):
            open_training_dataset(self.root)

        memmap_root = self.root / "memmap"
        memmap_root.mkdir()
        np.save(memmap_root / "X_t.npy", X_t)
        np.save(memmap_root / "delta.npy", np.zeros_like(X_t))
        np.save(memmap_root / "compression.npy", np.zeros((1, 1), dtype=np.float32))
        np.save(memmap_root / "theta.npy", np.zeros((1, 1), dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "offline extraction"):
            open_training_dataset(memmap_root)


class TrajectoryMetadataDataset(torch.utils.data.Dataset):
    """Splitting must need only small trajectory metadata, never point clouds."""

    def __init__(self, trajectory_ids: np.ndarray) -> None:
        self.trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.trajectory_ids)

    def metadata(self, name: str) -> np.ndarray:
        if name != "trajectory_id":
            raise KeyError(name)
        return self.trajectory_ids

    def __getitem__(self, index: int):
        raise AssertionError("Splitting must not load point clouds.")


class TrajectorySplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

        trajectory_ids = [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4]
        write_test_shard(
            self.root,
            0,
            list(range(8)),
            trajectory_ids[:8],
        )
        write_test_shard(
            self.root,
            1,
            list(range(8, len(trajectory_ids))),
            trajectory_ids[8:],
        )
        self.dataset = open_training_dataset(self.root)
        self.trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def split_ids(self, split: IndexedTransitionDataset) -> set[int]:
        return set(self.trajectory_ids[split.indices].tolist())

    def test_trajectory_split_has_no_group_leakage(self) -> None:
        train, validation, test, info = make_trajectory_train_val_test_datasets(
            self.dataset,
            val_fraction=0.25,
            seed=7,
            test_trajectory_id=4,
            test_fraction=0.4,
        )

        train_ids = self.split_ids(train)
        validation_ids = self.split_ids(validation)
        test_ids = self.split_ids(test)

        self.assertFalse(train_ids & validation_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(validation_ids & test_ids)
        self.assertEqual(train_ids | validation_ids | test_ids, {0, 1, 2, 3, 4})
        self.assertEqual(test_ids, {3, 4})  # Longest plus the requested additional ID.
        self.assertEqual(info.longest_trajectory_id, 3)
        self.assertEqual(len(train) + len(validation) + len(test), len(self.dataset))
        self.assertEqual(info.train_samples, len(train))
        self.assertEqual(info.validation_samples, len(validation))
        self.assertEqual(info.test_samples, len(test))

    def test_default_split_uses_trajectory_counts_and_forces_longest(self) -> None:
        ids = np.repeat(np.arange(20), np.arange(1, 21))
        dataset = TrajectoryMetadataDataset(ids)
        train, validation, test, info = make_trajectory_train_val_test_datasets(dataset, seed=7)
        self.assertEqual([len(info.train_trajectory_ids), len(info.validation_trajectory_ids),
                          len(info.test_trajectory_ids)], [16, 2, 2])
        self.assertEqual(info.longest_trajectory_id, 19)
        self.assertIn(19, info.test_trajectory_ids)
        # The forced longest occupies a test slot; it is not an extra third ID.
        self.assertEqual(np.count_nonzero(ids[test.indices] == 19), 20)
        groups = [set(ids[view.indices]) for view in (train, validation, test)]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        np.testing.assert_array_equal(
            np.sort(np.concatenate([train.indices, validation.indices, test.indices])),
            np.arange(len(dataset)),
        )

    def test_default_1847_group_counts_are_1477_185_185(self) -> None:
        dataset = TrajectoryMetadataDataset(np.r_[np.arange(1847), 738])
        info = make_trajectory_train_val_test_datasets(dataset, seed=7)[3]
        self.assertEqual([len(info.train_trajectory_ids), len(info.validation_trajectory_ids),
                          len(info.test_trajectory_ids)], [1477, 185, 185])
        self.assertEqual(info.longest_trajectory_id, 738)
        self.assertIn(738, info.test_trajectory_ids)

    def test_seed_changes_random_assignments_but_not_forced_longest(self) -> None:
        dataset = TrajectoryMetadataDataset(np.repeat(np.arange(20), np.arange(1, 21)))
        first = make_trajectory_train_val_test_datasets(dataset, seed=7)[3]
        self.assertEqual(first, make_trajectory_train_val_test_datasets(dataset, seed=7)[3])
        variants = [make_trajectory_train_val_test_datasets(dataset, seed=seed)[3]
                    for seed in range(8, 13)]
        # Different seeds can legitimately draw the same small subset. Check
        # that the seed affects selection, not that every pair must differ.
        self.assertTrue(any(first.test_trajectory_ids != item.test_trajectory_ids for item in variants))
        self.assertTrue(any(first.validation_trajectory_ids != item.validation_trajectory_ids for item in variants))
        self.assertTrue(all(19 in item.test_trajectory_ids for item in variants))

    def test_nonlongest_lengths_and_row_order_do_not_control_groups(self) -> None:
        lengths = np.r_[np.arange(1, 20), 100]
        ids = np.repeat(np.arange(20), lengths)
        reference = make_trajectory_train_val_test_datasets(TrajectoryMetadataDataset(ids), seed=7)[3]
        changed_lengths = np.r_[np.arange(19, 0, -1), 100]
        shuffled_ids = np.random.default_rng(123).permutation(np.repeat(np.arange(20), changed_lengths))
        changed = make_trajectory_train_val_test_datasets(
            TrajectoryMetadataDataset(shuffled_ids), seed=7,
        )[3]
        self.assertEqual(reference.train_trajectory_ids, changed.train_trajectory_ids)
        self.assertEqual(reference.validation_trajectory_ids, changed.validation_trajectory_ids)
        self.assertEqual(reference.test_trajectory_ids, changed.test_trajectory_ids)

    def test_tied_longest_and_tiny_corpus_are_deterministic(self) -> None:
        dataset = TrajectoryMetadataDataset(np.array([9, 4, 2]))
        train, validation, test, info = make_trajectory_train_val_test_datasets(dataset)
        self.assertEqual([len(train), len(validation), len(test)], [1, 1, 1])
        self.assertEqual(info.longest_trajectory_id, 2)
        self.assertEqual(info.test_trajectory_ids, (2,))

    def test_invalid_fractions_and_impossible_group_counts_fail(self) -> None:
        for name in ("val_fraction", "test_fraction"):
            for value in (0, -0.1, 1, float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    make_trajectory_train_val_test_datasets(self.dataset, **{name: value})
        with self.assertRaisesRegex(ValueError, "less than 1"):
            make_trajectory_train_val_test_datasets(self.dataset, val_fraction=0.5, test_fraction=0.5)
        small = TrajectoryMetadataDataset(np.arange(3))
        with self.assertRaisesRegex(ValueError, "no training trajectories"):
            make_trajectory_train_val_test_datasets(small, val_fraction=0.34, test_fraction=0.5)
        with self.assertRaisesRegex(ValueError, "At least three"):
            make_trajectory_train_val_test_datasets(TrajectoryMetadataDataset(np.arange(2)))

    def test_unknown_or_overfull_extra_test_id_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown test_trajectory_id"):
            make_trajectory_train_val_test_datasets(self.dataset, test_trajectory_id=999)
        with self.assertRaisesRegex(ValueError, "too few slots"):
            make_trajectory_train_val_test_datasets(self.dataset, test_trajectory_id=4)
        # Requesting the longest again must not consume another slot.
        info = make_trajectory_train_val_test_datasets(self.dataset, test_trajectory_id=3)[3]
        self.assertEqual(info.test_trajectory_ids, (3,))


if __name__ == "__main__":
    unittest.main()

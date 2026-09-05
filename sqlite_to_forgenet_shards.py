"""Extract a small ForgeNet-style training subset from large JAX-FEM SQLite DBs.

Run this script on the machine where the OneDrive `.db` files are already
available locally. It avoids copying the full database into this project and
writes small `.npz` shards that the notebook can stream.

The expected database layout follows the public OSU-SIMCenter forge-net data
processor: a `strike` table with at least `series_id`, `result`, `position`, and
`rotation` columns. The `result` column is JSON containing `Steps`, `Vertices`,
and usually `Triangles`.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


@dataclass
class ExtractionStats:
    rows_read: int = 0
    transitions_written: int = 0
    skipped_bad_json: int = 0
    skipped_bad_vertices: int = 0
    skipped_topology_mismatch: int = 0
    skipped_too_few_points: int = 0
    skipped_no_previous_state: int = 0


def parse_json_field(value: object, default: object | None = None) -> object:
    """Parse a SQLite JSON/text field."""

    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def final_vertices(result_json: dict) -> np.ndarray:
    """Extract final-step vertices from a forge-net/JAX-FEM result JSON object."""

    steps = result_json.get("Steps", [])
    vertices = np.asarray(result_json["Vertices"], dtype=np.float32)
    num_steps = max(1, len(steps))
    vertices = vertices.reshape(num_steps, -1)[-1].reshape(-1, 3)
    return vertices.astype(np.float32, copy=False)


def compression_from_result(result_json: dict) -> float:
    """Use the sum of the next row's Steps as the scalar compression action."""

    steps = np.asarray(result_json.get("Steps", []), dtype=np.float32)
    if steps.size == 0:
        return 0.0
    return float(np.sum(steps))


def quat_rotate(points: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Rotate points by a quaternion in SciPy/PyVista `[x, y, z, w]` order."""

    q = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    x, y, z, w = q
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return points
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    # Rotation matrix matching scipy.spatial.transform.Rotation.from_quat.
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return points @ R.T


def transform_points(points: np.ndarray, rotation: object, position: object) -> np.ndarray:
    """Apply forge-net's `transform_points(points, rotation, position)` behavior."""

    quat = np.asarray(rotation, dtype=np.float32)
    shift = np.asarray(position, dtype=np.float32)
    if quat.shape != (4,) or shift.shape != (3,):
        return points
    return quat_rotate(points, quat) + shift


def iter_rows(
    db_path: Path,
    table: str,
    line_limit: int | None,
    order_by_series: bool,
    series_ids: set[str] | None = None,
) -> Iterator[sqlite3.Row]:
    """Yield rows from a SQLite database."""

    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        order_clause = "ORDER BY series_id, rowid" if order_by_series else "ORDER BY rowid"
        where_clause = ""
        query_params: list[object] = []
        if series_ids:
            placeholders = ",".join("?" for _ in series_ids)
            where_clause = f"WHERE series_id IN ({placeholders})"
            query_params.extend(sorted(series_ids))
        limit_clause = "" if line_limit is None else "LIMIT ?"
        query = (
            f"SELECT rowid, series_id, result, position, rotation "
            f"FROM {table} {where_clause} {order_clause} {limit_clause}"
        )
        if line_limit is not None:
            query_params.append(int(line_limit))
        cursor = conn.execute(query, tuple(query_params))
        yield from cursor
    finally:
        conn.close()


def choose_point_indices(
    num_points: int,
    points_per_state: int | None,
    rng: np.random.Generator,
) -> np.ndarray | slice:
    """Choose point identities to keep."""

    if points_per_state is None:
        return slice(None)
    if num_points < points_per_state:
        raise ValueError("not enough points")
    return np.sort(rng.choice(num_points, size=points_per_state, replace=False))


def flush_shard(
    output_dir: Path,
    shard_idx: int,
    X_t_buffer: list[np.ndarray],
    delta_buffer: list[np.ndarray],
    compression_buffer: list[list[float]],
    trajectory_buffer: list[int],
    trajectory_key_buffer: list[str],
    step_buffer: list[int],
    source_buffer: list[str],
    position_buffer: list[np.ndarray],
    rotation_buffer: list[np.ndarray],
) -> Path:
    """Write one `.npz` shard and clear caller-owned buffers afterward."""

    X_t = np.stack(X_t_buffer).astype(np.float32)
    delta = np.stack(delta_buffer).astype(np.float32)
    compression = np.asarray(compression_buffer, dtype=np.float32)
    X_next = X_t + delta
    path = output_dir / f"shard_{shard_idx:05d}.npz"
    np.savez_compressed(
        path,
        X_t=X_t,
        delta=delta,
        compression=compression,
        X_next=X_next.astype(np.float32),
        trajectory_id=np.asarray(trajectory_buffer, dtype=np.int64),
        trajectory_key=np.asarray(trajectory_key_buffer),
        step_id=np.asarray(step_buffer, dtype=np.int64),
        source_db=np.asarray(source_buffer),
        position=np.stack(position_buffer).astype(np.float32),
        rotation=np.stack(rotation_buffer).astype(np.float32),
    )
    return path


def extract_db(
    db_path: Path,
    output_dir: Path,
    table: str,
    max_transitions: int,
    line_limit: int | None,
    samples_per_shard: int,
    points_per_state: int | None,
    seed: int,
    apply_pose: bool,
    order_by_series: bool,
    series_ids: set[str] | None = None,
    starting_shard_idx: int = 0,
    starting_trajectory_idx: int = 0,
) -> tuple[ExtractionStats, int, int, list[Path]]:
    """Extract transitions from one SQLite database."""

    stats = ExtractionStats()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    prev_by_series: dict[str, sqlite3.Row] = {}
    trajectory_id_by_key: dict[str, int] = {}
    point_indices_by_series: dict[str, np.ndarray | slice] = {}
    vertex_count_by_series: dict[str, int] = {}
    shard_paths: list[Path] = []
    shard_idx = starting_shard_idx
    X_t_buffer: list[np.ndarray] = []
    delta_buffer: list[np.ndarray] = []
    compression_buffer: list[list[float]] = []
    trajectory_buffer: list[int] = []
    trajectory_key_buffer: list[str] = []
    step_buffer: list[int] = []
    source_buffer: list[str] = []
    position_buffer: list[np.ndarray] = []
    rotation_buffer: list[np.ndarray] = []

    def maybe_flush(force: bool = False) -> None:
        nonlocal shard_idx
        if not X_t_buffer:
            return
        if not force and len(X_t_buffer) < samples_per_shard:
            return
        path = flush_shard(
            output_dir=output_dir,
            shard_idx=shard_idx,
            X_t_buffer=X_t_buffer,
            delta_buffer=delta_buffer,
            compression_buffer=compression_buffer,
            trajectory_buffer=trajectory_buffer,
            trajectory_key_buffer=trajectory_key_buffer,
            step_buffer=step_buffer,
            source_buffer=source_buffer,
            position_buffer=position_buffer,
            rotation_buffer=rotation_buffer,
        )
        shard_paths.append(path)
        shard_idx += 1
        X_t_buffer.clear()
        delta_buffer.clear()
        compression_buffer.clear()
        trajectory_buffer.clear()
        trajectory_key_buffer.clear()
        step_buffer.clear()
        source_buffer.clear()
        position_buffer.clear()
        rotation_buffer.clear()

    for row in iter_rows(
        db_path,
        table=table,
        line_limit=line_limit,
        order_by_series=order_by_series,
        series_ids=series_ids,
    ):
        stats.rows_read += 1
        series_key = str(row["series_id"])
        if series_key not in trajectory_id_by_key:
            trajectory_id_by_key[series_key] = starting_trajectory_idx + len(trajectory_id_by_key)
        trajectory_id = trajectory_id_by_key[series_key]

        prev = prev_by_series.get(series_key)
        prev_by_series[series_key] = row
        if prev is None:
            stats.skipped_no_previous_state += 1
            continue

        try:
            result_t = parse_json_field(prev["result"])
            result_tp1 = parse_json_field(row["result"])
            X_t_full = final_vertices(result_t)
            X_tp1_full = final_vertices(result_tp1)
            position_tp1 = parse_json_field(row["position"], default=None)
            rotation_tp1 = parse_json_field(row["rotation"], default=None)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            stats.skipped_bad_json += 1
            continue

        position_tp1 = np.asarray(position_tp1, dtype=np.float32)
        rotation_tp1 = np.asarray(rotation_tp1, dtype=np.float32)
        if position_tp1.shape != (3,) or rotation_tp1.shape != (4,):
            stats.skipped_bad_json += 1
            continue

        if X_t_full.ndim != 2 or X_tp1_full.ndim != 2:
            stats.skipped_bad_vertices += 1
            continue
        if X_t_full.shape != X_tp1_full.shape or X_t_full.shape[-1] != 3:
            stats.skipped_topology_mismatch += 1
            continue

        if apply_pose:
            X_t_full = transform_points(X_t_full, rotation_tp1, position_tp1)
            X_tp1_full = transform_points(X_tp1_full, rotation_tp1, position_tp1)

        if series_key not in point_indices_by_series:
            try:
                point_indices_by_series[series_key] = choose_point_indices(
                    X_t_full.shape[0],
                    points_per_state=points_per_state,
                    rng=rng,
                )
            except ValueError:
                stats.skipped_too_few_points += 1
                continue
            vertex_count_by_series[series_key] = X_t_full.shape[0]
        elif vertex_count_by_series[series_key] != X_t_full.shape[0]:
            stats.skipped_topology_mismatch += 1
            continue
        point_idx = point_indices_by_series[series_key]

        X_t = X_t_full[point_idx].astype(np.float32)
        X_tp1 = X_tp1_full[point_idx].astype(np.float32)
        delta = X_tp1 - X_t
        compression = compression_from_result(result_tp1)

        X_t_buffer.append(X_t)
        delta_buffer.append(delta.astype(np.float32))
        compression_buffer.append([compression])
        trajectory_buffer.append(trajectory_id)
        trajectory_key_buffer.append(series_key)
        step_buffer.append(int(row["rowid"]))
        source_buffer.append(db_path.name)
        position_buffer.append(position_tp1)
        rotation_buffer.append(rotation_tp1)
        stats.transitions_written += 1

        maybe_flush(force=False)
        if stats.transitions_written >= max_transitions:
            break

    maybe_flush(force=True)
    next_trajectory_idx = starting_trajectory_idx + len(trajectory_id_by_key)
    return stats, shard_idx, next_trajectory_idx, shard_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a small processed ForgeNet subset from large SQLite DBs."
    )
    parser.add_argument("db", nargs="+", type=Path, help="Path(s) to hydrated .db files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/jax_fem_shards_small"),
        help="Directory for output shard_*.npz files.",
    )
    parser.add_argument("--table", default="strike", help="SQLite table name.")
    parser.add_argument(
        "--max-transitions",
        type=int,
        default=500,
        help="Maximum transitions to write per database.",
    )
    parser.add_argument(
        "--line-limit",
        type=int,
        default=10000,
        help="Maximum rows to scan per database. Use 0 for no limit.",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=128,
        help="Number of transitions per shard.",
    )
    parser.add_argument(
        "--points-per-state",
        type=int,
        default=1024,
        help="Random vertex identities per state. Use 0 to keep all vertices.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for point subsampling.")
    parser.add_argument(
        "--no-apply-pose",
        action="store_true",
        help="Do not transform both states using the next row's position/rotation.",
    )
    parser.add_argument(
        "--order-by-series",
        action="store_true",
        help="Order rows by series_id,rowid. More robust, but can be slower on huge DBs.",
    )
    parser.add_argument(
        "--series-ids-file",
        type=Path,
        default=None,
        help=(
            "Optional text file containing one series_id per line. Only complete rows "
            "from those trajectories are scanned."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    line_limit = None if args.line_limit == 0 else args.line_limit
    points_per_state = None if args.points_per_state == 0 else args.points_per_state
    series_ids = None
    if args.series_ids_file is not None:
        series_ids = {
            line.strip()
            for line in args.series_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        if not series_ids:
            raise ValueError(f"No series IDs found in {args.series_ids_file}.")

    all_stats: dict[str, dict[str, int]] = {}
    shard_idx = 0
    trajectory_idx = 0
    all_shards: list[str] = []
    for db_path in args.db:
        stats, shard_idx, trajectory_idx, shard_paths = extract_db(
            db_path=db_path,
            output_dir=args.output_dir,
            table=args.table,
            max_transitions=args.max_transitions,
            line_limit=line_limit,
            samples_per_shard=args.samples_per_shard,
            points_per_state=points_per_state,
            seed=args.seed + shard_idx,
            apply_pose=not args.no_apply_pose,
            order_by_series=args.order_by_series,
            series_ids=series_ids,
            starting_shard_idx=shard_idx,
            starting_trajectory_idx=trajectory_idx,
        )
        all_stats[str(db_path)] = asdict(stats)
        all_shards.extend(str(path) for path in shard_paths)
        print(f"{db_path}: wrote {stats.transitions_written} transitions")

    metadata = {
        "db_paths": [str(path) for path in args.db],
        "output_dir": str(args.output_dir),
        "points_per_state": points_per_state,
        "samples_per_shard": args.samples_per_shard,
        "line_limit": line_limit,
        "max_transitions_per_db": args.max_transitions,
        "apply_pose": not args.no_apply_pose,
        "order_by_series": args.order_by_series,
        "series_ids_file": None if args.series_ids_file is None else str(args.series_ids_file),
        "selected_series_count": None if series_ids is None else len(series_ids),
        "stats": all_stats,
        "shards": all_shards,
        "fields": [
            "X_t",
            "delta",
            "compression",
            "X_next",
            "trajectory_id",
            "trajectory_key",
            "step_id",
            "source_db",
            "position",
            "rotation",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "extraction_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata to {args.output_dir / 'extraction_metadata.json'}")
    print(f"Notebook config.data_path can now point to: {args.output_dir}")


if __name__ == "__main__":
    main()

"""Initial billet surface sampling utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np


def sample_initial_billet(
    R0: float,
    H0: float,
    N: int | None = None,
    rotate: bool = True,
    seed: Optional[int] = None,
    return_labels: bool = False,
    *,
    target_points: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Sample a cylindrical billet surface point cloud.

    Parameters
    ----------
    R0:
        Initial cylinder radius.
    H0:
        Initial cylinder half-height. The top cap is at ``+H0`` and the bottom
        cap is at ``-H0``.
    N:
        Total number of surface points. Points are allocated between
        the side wall and two caps according to their surface areas.
        Provide either ``N`` or ``target_points``, not both.
    rotate:
        If true, apply a random z-axis rotation after sampling.
    seed:
        Optional NumPy random seed.
    return_labels:
        If true, also return a string label for each point: ``side``,
        ``top_cap``, or ``bottom_cap``.
    target_points:
        Optional nonempty target point cloud with shape ``(M, 3)``. When
        provided, use ``M`` as ``N``. Only the number of points is used;
        target coordinates do not change the billet sampling method.

    Returns
    -------
    X0:
        Array with shape ``(N, 3)``.
    labels:
        Optional array with one semantic surface label per point.
    """

    if R0 <= 0:
        raise ValueError("R0 must be positive.")
    if H0 <= 0:
        raise ValueError("H0 must be positive.")
    if target_points is not None:
        if N is not None:
            raise ValueError("Provide either N or target_points, not both.")
        target_points = np.asarray(target_points)
        if (
            target_points.ndim != 2
            or target_points.shape[1] != 3
            or target_points.shape[0] == 0
        ):
            raise ValueError("target_points must have nonempty shape (M, 3).")
        N = int(target_points.shape[0])
    if N is None:
        raise ValueError("Provide N or target_points.")
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)):
        raise TypeError("N must be an integer.")
    if N <= 0:
        raise ValueError("N must be positive.")

    rng = np.random.default_rng(seed)

    n_cap_each = round(
        N * R0 / (2.0 * (2.0 * H0 + R0))
    )

    N_caps = 2 * n_cap_each
    N_side = N - N_caps

    alpha_side = rng.uniform(0.0, 2.0 * np.pi, size=N_side)
    z_side = rng.uniform(-H0, H0, size=N_side)
    side = np.column_stack(
        [
            R0 * np.cos(alpha_side),
            R0 * np.sin(alpha_side),
            z_side,
        ]
    )

    n_cap_each = N_caps // 2

    def sample_cap(z_value: float) -> np.ndarray:
        u = rng.uniform(0.0, 1.0, size=n_cap_each)
        r = R0 * np.sqrt(u)
        alpha = rng.uniform(0.0, 2.0 * np.pi, size=n_cap_each)
        return np.column_stack(
            [
                r * np.cos(alpha),
                r * np.sin(alpha),
                np.full(n_cap_each, z_value, dtype=np.float64),
            ]
        )

    top = sample_cap(+H0)
    bottom = sample_cap(-H0)
    X0 = np.vstack([side, top, bottom])

    if rotate:
        gamma = rng.uniform(0.0, 2.0 * np.pi)
        c, s = np.cos(gamma), np.sin(gamma)
        R_z = np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        X0 = X0 @ R_z.T

    X0 = X0.astype(np.float32)

    if not return_labels:
        return X0

    labels = np.concatenate(
        [
            np.full(N_side, "side", dtype=object),
            np.full(n_cap_each, "top_cap", dtype=object),
            np.full(n_cap_each, "bottom_cap", dtype=object),
        ]
    )
    return X0, labels

"""ForgeNet model adapted from the OSU-SIMCenter forge-net repository."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ForgeNet(nn.Module):
    """Repo-style ForgeNet for point-cloud transition deltas.

    This follows the architecture used in OSU-SIMCenter/forge-net:

    - channels-first point-cloud input ``x_t`` with shape ``(B, 3, N)``
    - action input ``a_t`` with shape ``(B, action_dims)``
    - shared 1x1 state encoder, max pooling, action MLP, tiled global latent
    - pointwise decoder returning a scaled displacement with shape ``(B, N, 3)``

    The network predicts the scaled delta used by the original training
    process. Convert it back to physical displacement with
    ``predict_physical_delta``.
    """

    def __init__(
        self,
        point_size: int,
        latent_size: int,
        action_dims: int,
        dropout: float = 0.3,
        use_res: bool = True,
        delta_scalar: float = 100.0,
    ) -> None:
        super().__init__()
        self.latent_size = int(latent_size / 2)
        self.point_size = point_size
        self.action_dims = action_dims
        self.dropout = dropout
        self.use_res = use_res
        self.delta_scalar = float(delta_scalar)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.proj1 = nn.Conv1d(64, 128, 1)
        self.bn_proj1 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 256, 1)
        self.bn4 = nn.BatchNorm1d(256)
        self.proj2 = nn.Conv1d(128, 256, 1)
        self.bn_proj2 = nn.BatchNorm1d(256)

        self.conv5 = nn.Conv1d(256, self.latent_size, 1)
        self.bn5 = nn.BatchNorm1d(self.latent_size)
        self.dropout_state = nn.Dropout(dropout)

        self.act_fc1 = nn.Linear(action_dims, 64)
        self.act_bn1 = nn.BatchNorm1d(64)
        self.act_fc2 = nn.Linear(64, 64)
        self.act_bn2 = nn.BatchNorm1d(64)
        self.act_fc3 = nn.Linear(64, 128)
        self.act_bn3 = nn.BatchNorm1d(128)
        self.act_proj1 = nn.Linear(64, 128)
        self.act_bn_proj1 = nn.BatchNorm1d(128)
        self.act_fc4 = nn.Linear(128, 256)
        self.act_bn4 = nn.BatchNorm1d(256)
        self.act_proj2 = nn.Linear(128, 256)
        self.act_bn_proj2 = nn.BatchNorm1d(256)
        self.act_fc5 = nn.Linear(256, self.latent_size)
        self.dropout_action = nn.Dropout(dropout)

        decoder_in_dim = (self.latent_size * 2) + 3
        self.dec_conv1 = nn.Conv1d(decoder_in_dim, 512, 1)
        self.dec_bn1 = nn.BatchNorm1d(512)
        self.dec_conv2 = nn.Conv1d(512, 256, 1)
        self.dec_bn2 = nn.BatchNorm1d(256)
        self.dec_conv3 = nn.Conv1d(256, 128, 1)
        self.dec_bn3 = nn.BatchNorm1d(128)
        self.dec_proj1 = nn.Conv1d(512, 128, 1)
        self.dec_bn_proj1 = nn.BatchNorm1d(128)
        self.dec_conv4 = nn.Conv1d(128, 3, 1)
        self.dropout_dec = nn.Dropout(dropout * 0.5)

        self._build_codecs()
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _build_codecs(self) -> None:
        if self.use_res:

            def state_encoder(x: torch.Tensor) -> torch.Tensor:
                x = F.relu(self.bn1(self.conv1(x)))
                identity = x
                x = F.relu(self.bn2(self.conv2(x))) + identity
                identity = self.bn_proj1(self.proj1(x))
                x = F.relu(self.bn3(self.conv3(x))) + identity
                identity = self.bn_proj2(self.proj2(x))
                x = F.relu(self.bn4(self.conv4(x))) + identity
                x = self.bn5(self.conv5(x))
                x = torch.max(x, 2, keepdim=False)[0]
                return self.dropout_state(x)

            def action_encoder(a: torch.Tensor) -> torch.Tensor:
                if len(a.shape) == 1:
                    a = a.unsqueeze(1)
                a = F.relu(self.act_bn1(self.act_fc1(a)))
                identity = a
                a = F.relu(self.act_bn2(self.act_fc2(a))) + identity
                identity = self.act_bn_proj1(self.act_proj1(a))
                a = F.relu(self.act_bn3(self.act_fc3(a))) + identity
                identity = self.act_bn_proj2(self.act_proj2(a))
                a = F.relu(self.act_bn4(self.act_fc4(a))) + identity
                return self.act_fc5(self.dropout_action(a))

            def decoder(combined_features: torch.Tensor) -> torch.Tensor:
                x = F.relu(self.dec_bn1(self.dec_conv1(combined_features)))
                x = self.dropout_dec(x)
                identity = self.dec_bn_proj1(self.dec_proj1(x))
                x = F.relu(self.dec_bn2(self.dec_conv2(x)))
                x = F.relu(self.dec_bn3(self.dec_conv3(x))) + identity
                delta = self.dec_conv4(x)
                return delta.transpose(1, 2)

        else:

            def state_encoder(x: torch.Tensor) -> torch.Tensor:
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.relu(self.bn3(self.conv3(x)))
                x = F.relu(self.bn4(self.conv4(x)))
                x = self.bn5(self.conv5(x))
                x = torch.max(x, 2, keepdim=False)[0]
                return self.dropout_state(x)

            def action_encoder(a: torch.Tensor) -> torch.Tensor:
                if len(a.shape) == 1:
                    a = a.unsqueeze(1)
                a = F.relu(self.act_bn1(self.act_fc1(a)))
                a = F.relu(self.act_bn2(self.act_fc2(a)))
                a = F.relu(self.act_bn3(self.act_fc3(a)))
                a = F.relu(self.act_bn4(self.act_fc4(a)))
                return self.act_fc5(self.dropout_action(a))

            def decoder(combined_features: torch.Tensor) -> torch.Tensor:
                x = F.relu(self.dec_bn1(self.dec_conv1(combined_features)))
                x = self.dropout_dec(x)
                x = F.relu(self.dec_bn2(self.dec_conv2(x)))
                x = F.relu(self.dec_bn3(self.dec_conv3(x)))
                delta = self.dec_conv4(x)
                return delta.transpose(1, 2)

        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder

    def forward(self, x_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """Predict scaled per-point deltas from channels-first point clouds."""

        if x_t.ndim != 3 or x_t.shape[1] != 3:
            raise ValueError(f"x_t must have shape (B, 3, N); received {tuple(x_t.shape)}.")
        if a_t.ndim == 1:
            a_t = a_t.unsqueeze(1)
        if a_t.ndim != 2 or a_t.shape[1] != self.action_dims:
            raise ValueError(
                f"a_t must have shape (B, {self.action_dims}); received {tuple(a_t.shape)}."
            )

        B, _, N = x_t.shape
        x_l = self.state_encoder(x_t)
        a_l = self.action_encoder(a_t)
        global_latent = torch.cat([x_l, a_l], dim=1)
        global_expanded = global_latent.unsqueeze(2).expand(-1, -1, N)
        combined_features = torch.cat([global_expanded, x_t], dim=1)
        return self.decoder(combined_features)


def _one_step_action(action: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Normalize one action per transition to shape ``(B, action_dims)``."""

    if action.ndim == 1:
        normalized = action[:, None]
    elif action.ndim == 2:
        normalized = action
    elif action.ndim == 3:
        if action.shape[1] != 1:
            raise ValueError(
                "ForgeNet accepts one action per transition; "
                f"received {action.shape[1]} actions with shape {tuple(action.shape)}. "
                "Roll out action sequences one step at a time."
            )
        normalized = action[:, 0, :]
    else:
        raise ValueError(
            "compression_action must have shape (B,), (B, action_dims), or "
            f"(B, 1, action_dims); received {tuple(action.shape)}."
        )

    if normalized.shape[0] != batch_size:
        raise ValueError(
            "Point-cloud and action batch sizes must match; "
            f"received {batch_size} point clouds and {normalized.shape[0]} actions."
        )
    return normalized


def predict_scaled_delta(
    model: nn.Module,
    X: torch.Tensor,
    compression_action: torch.Tensor,
) -> torch.Tensor:
    """Call ForgeNet from notebook-friendly ``(B, N, 3)`` inputs."""

    if X.ndim != 3 or X.shape[-1] != 3:
        raise ValueError(f"X must have shape (B, N, 3); received {tuple(X.shape)}.")
    compression_action = _one_step_action(compression_action, batch_size=X.shape[0])
    return model(X.permute(0, 2, 1), compression_action)


def predict_physical_delta(
    model: nn.Module,
    X: torch.Tensor,
    compression_action: torch.Tensor,
    delta_scalar: float | None = None,
) -> torch.Tensor:
    """Return unscaled physical displacement for rollout and metrics."""

    if delta_scalar is None:
        delta_scalar = float(getattr(model, "delta_scalar", 1.0))
    action = _one_step_action(compression_action, batch_size=X.shape[0])
    physical_delta = predict_scaled_delta(model, X, action) / float(delta_scalar)

    # A zero die closure is a true hold action. Enforcing that boundary condition
    # prevents accumulated network bias from moving the billet after control stops.
    active = torch.any(torch.abs(action) > 1.0e-12, dim=1).to(physical_delta.dtype)
    return physical_delta * active[:, None, None]
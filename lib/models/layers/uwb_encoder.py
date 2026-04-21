import torch.nn as nn


class UWBMLPEncoder(nn.Module):
    """Encode a UWB (u, v) sequence into a 768-d feature/token with an MLP."""

    def __init__(
        self,
        in_dim=2,
        hidden_dims=(32, 64, 128, 256),
        out_dim=768,
        temporal_pool="mean",
    ):
        super().__init__()
        if temporal_pool not in ["mean", "max"]:
            raise ValueError("temporal_pool must be one of ['mean', 'max']")

        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.frame_mlp = nn.Sequential(*layers)
        self.temporal_pool = temporal_pool

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, uwb_uv):
        """
        Args:
            uwb_uv: [B, T, 2], UWB (u, v) sequence in image coordinates.
        Returns:
            seq_feat: [B, 1, 768], pooled sequence feature in token format.
        """
        if uwb_uv.ndim != 3 or uwb_uv.size(-1) != 2:
            raise ValueError("uwb_uv must have shape [B, T, 2]")

        bsz, t, _ = uwb_uv.shape
        x = uwb_uv.reshape(bsz * t, -1)
        frame_feat = self.frame_mlp(x).reshape(bsz, t, -1)

        if self.temporal_pool == "mean":
            pooled_feat = frame_feat.mean(dim=1)
        else:
            pooled_feat = frame_feat.max(dim=1)[0]

        seq_feat = pooled_feat.unsqueeze(1)
        return seq_feat


class UWBConv1DEncoder(nn.Module):
    """Encode a UWB (u, v) sequence with temporal 1D convolutions."""

    def __init__(
        self,
        in_dim=2,
        channels=(32, 64, 128, 256),
        out_dim=768,
        kernel_size=3,
        temporal_pool="mean",
    ):
        super().__init__()
        if temporal_pool not in ["mean", "max"]:
            raise ValueError("temporal_pool must be one of ['mean', 'max']")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        dims = [in_dim] + list(channels)
        layers = []
        padding = kernel_size // 2
        for i in range(len(dims) - 1):
            layers.append(nn.Conv1d(dims[i], dims[i + 1], kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*layers)
        self.proj = nn.Linear(dims[-1], out_dim)
        self.temporal_pool = temporal_pool

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, uwb_uv):
        """
        Args:
            uwb_uv: [B, T, 2], UWB (u, v) sequence in image coordinates.
        Returns:
            seq_feat: [B, 1, 768], pooled sequence feature in token format.
        """
        if uwb_uv.ndim != 3 or uwb_uv.size(-1) != 2:
            raise ValueError("uwb_uv must have shape [B, T, 2]")

        x = uwb_uv.transpose(1, 2)
        frame_feat = self.temporal_conv(x).transpose(1, 2)

        if self.temporal_pool == "mean":
            pooled_feat = frame_feat.mean(dim=1)
        else:
            pooled_feat = frame_feat.max(dim=1)[0]

        seq_feat = self.proj(pooled_feat).unsqueeze(1)
        return seq_feat

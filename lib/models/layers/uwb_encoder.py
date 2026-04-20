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

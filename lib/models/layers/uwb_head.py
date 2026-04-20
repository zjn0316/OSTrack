import torch.nn as nn


class UWBHead(nn.Module):
    """Task head: 768 -> 256 -> 64 -> task_dim."""

    def __init__(self, in_dim=768, hidden_dims=(256, 64), task_dim=2, final_act=None):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [task_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

        if final_act is None:
            self.final_act = nn.Identity()
        elif final_act == "sigmoid":
            self.final_act = nn.Sigmoid()
        elif final_act == "tanh":
            self.final_act = nn.Tanh()
        else:
            raise ValueError("final_act must be one of [None, 'sigmoid', 'tanh']")

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: [B, 768] or [B, 1, 768]
        Returns:
            y: [B, task_dim]
        """
        if x.ndim == 3:
            if x.size(1) != 1:
                raise ValueError("If x is 3D, expected shape [B, 1, 768]")
            x = x[:, 0, :]
        return self.final_act(self.mlp(x))


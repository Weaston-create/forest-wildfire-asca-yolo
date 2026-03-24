import torch
import torch.nn as nn


class FWSCSAttention(nn.Module):
    """
    Forest Wildfire Sparse Contextual Saliency Attention (FWSCSAttention)
    森林野火稀疏上下文显著性感知注意力机制
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        # -------- 1. Context statistics --------
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(2, 3), keepdim=True)
        std = torch.sqrt(var + self.eps)

        # -------- 2. Sparse deviation modeling --------
        deviation = torch.abs(x - mean) / (std + self.eps)

        # -------- 3. Adaptive saliency gate (no params) --------
        gate = torch.tanh(deviation)

        # -------- 4. Residual enhancement --------
        out = x + gate * deviation * x

        return out
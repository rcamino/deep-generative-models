import torch

from torch import Tensor
from torch.nn import Module
from torch.nn.functional import mse_loss


class RMSE(Module):
    reduction: str

    def __init__(self, reduction: str = "mean"):
        super(RMSE, self).__init__()
        self.reduction = reduction

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        return torch.sqrt(mse_loss(inputs, target, reduction=self.reduction))

import torch

from torch import Tensor
from torch.nn import Module


class AdditiveNormalNoise(Module):
    noise_mean: float
    noise_std: float

    def __init__(self, noise_mean: float = 0, noise_std: float = 1):
        super(AdditiveNormalNoise, self).__init__()
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.empty_like(inputs).normal_(self.noise_mean, self.noise_std) + inputs

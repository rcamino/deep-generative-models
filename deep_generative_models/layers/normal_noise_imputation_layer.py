import torch

from torch import Tensor

from deep_generative_models.imputation.masks import compose_with_mask
from deep_generative_models.layers.imputation_layer import ImputationLayer


class NormalNoiseImputation(ImputationLayer):
    noise_mean: float
    noise_std: float
    differentiable: bool

    def __init__(self, noise_mean: float = 0, noise_std: float = 1, differentiable: bool = True) -> None:
        super(NormalNoiseImputation, self).__init__()
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.differentiable = differentiable

    def forward(self, inputs: Tensor, missing_mask: Tensor) -> Tensor:
        return compose_with_mask(missing_mask,
                                 where_one=torch.empty_like(inputs).normal_(self.noise_mean, self.noise_std),
                                 where_zero=inputs,
                                 differentiable=self.differentiable)

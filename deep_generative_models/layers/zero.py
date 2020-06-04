import torch

from torch import Tensor

from deep_generative_models.imputation.masks import compose_with_mask
from deep_generative_models.layers.imputation_layer import ImputationLayer


class ZeroImputation(ImputationLayer):
    differentiable: bool
    
    def __init__(self, differentiable: bool = True) -> None:
        super(ZeroImputation, self).__init__()
        self.differentiable = differentiable
    
    def forward(self, inputs: Tensor, missing_mask: Tensor) -> Tensor:
        return compose_with_mask(missing_mask,
                                 where_one=torch.zeros_like(inputs),
                                 where_zero=inputs,
                                 differentiable=self.differentiable)

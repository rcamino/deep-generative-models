import torch

from typing import Any, List

from torch import Tensor
from torch.nn import Module

from deep_generative_models.architecture import Architecture
from deep_generative_models.component_factory import MultiComponentFactory
from deep_generative_models.configuration import Configuration
from deep_generative_models.gpu import to_gpu_if_available
from deep_generative_models.imputation.masks import compose_with_mask, generate_mask_for
from deep_generative_models.metadata import Metadata


class MultiInputDropout(Module):
    metadata: Metadata
    drop_probability: float

    def __init__(self, metadata: Metadata, drop_probability: float = 0.5):
        super(MultiInputDropout, self).__init__()
        self.metadata = metadata
        self.drop_probability = drop_probability

    def forward(self, inputs: Tensor) -> Tensor:
        # dropout only during training
        if self.training:
            # create a missing mask using the drop probability
            drop_mask = to_gpu_if_available(generate_mask_for(inputs, self.drop_probability, self.metadata))

            # put zeros where the drop mask is one and leave the inputs where the drop mask is zero
            return compose_with_mask(mask=drop_mask,
                                     where_one=torch.zeros_like(inputs),
                                     where_zero=inputs,
                                     differentiable=True)

        # don't touch the inputs during evaluation
        else:
            return inputs


class MultiInputDropoutFactory(MultiComponentFactory):

    def optional_arguments(self) -> List[str]:
        return ["drop_probability"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        return MultiInputDropout(metadata, **arguments.get_all_defined(self.optional_arguments()))

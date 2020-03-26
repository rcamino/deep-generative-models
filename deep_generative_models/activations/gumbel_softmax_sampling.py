from typing import Any

from torch import Tensor
from torch.nn import Module

from torch.nn.functional import gumbel_softmax

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.factory import Factory
from deep_generative_models.metadata import Metadata


class GumbelSoftmaxSampling(Module):
    temperature: float

    def __init__(self, temperature: float) -> None:
        super(GumbelSoftmaxSampling, self).__init__()
        self.temperature = temperature

    def forward(self, inputs: Tensor) -> Tensor:
        return gumbel_softmax(inputs, hard=not self.training, tau=self.temperature)


class GumbelSoftmaxSamplingFactory(Factory):

    def create(self, architecture: Architecture, metadata: Metadata, global_configuration: Configuration,
               configuration: Configuration) -> Any:
        return GumbelSoftmaxSampling(global_configuration.temperature)

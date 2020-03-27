from typing import Any

from torch import Tensor

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.layers.input_layer import InputLayer
from deep_generative_models.metadata import Metadata
from deep_generative_models.factory import MultiFactory


class SingleInputLayer(InputLayer):
    output_size: int

    def __init__(self, input_size: int) -> None:
        super(SingleInputLayer, self).__init__()
        self.output_size = input_size

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs

    def get_output_size(self) -> int:
        return self.output_size


class SingleInputLayerFactory(MultiFactory):

    def create(self, architecture: Architecture, metadata: Metadata, global_configuration: Configuration,
               configuration: Configuration) -> Any:
        return SingleInputLayer(configuration.input_size)

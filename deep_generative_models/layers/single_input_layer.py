from typing import Any, List, Optional

from torch import Tensor

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.layers.input_layer import InputLayer
from deep_generative_models.metadata import Metadata
from deep_generative_models.component_factory import ComponentFactory


class SingleInputLayer(InputLayer):
    output_size: int

    def __init__(self, input_size: int) -> None:
        super(SingleInputLayer, self).__init__()
        self.output_size = input_size

    def forward(self, inputs: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        if condition is not None:
            raise Exception("An unexpected condition was received.")
        return inputs

    def get_output_size(self) -> int:
        return self.output_size


class SingleInputLayerFactory(ComponentFactory):

    def mandatory_arguments(self) -> List[str]:
        return ["input_size"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        return SingleInputLayer(arguments.input_size)

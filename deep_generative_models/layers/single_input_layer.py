from typing import Any, List

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

    def forward(self, inputs: Tensor, **additional_inputs: Tensor) -> Tensor:
        # this is a "leaf" input layer (no child input layers)
        # so no additional inputs should remain
        for additional_inputs_name, additional_inputs_value in additional_inputs.items():
            if additional_inputs_value is not None:  # sometimes it makes things easier if I pass None
                raise Exception("Unexpected additional inputs received: {}.".format(additional_inputs_name))
        return inputs

    def get_output_size(self) -> int:
        return self.output_size


class SingleInputLayerFactory(ComponentFactory):

    def mandatory_arguments(self) -> List[str]:
        return ["input_size"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        return SingleInputLayer(arguments.input_size)

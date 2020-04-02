from torch import Tensor
from torch.nn import Module, Linear, Sequential

from typing import Optional, Any, List

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.layers.output_layer import OutputLayerFactory
from deep_generative_models.metadata import Metadata
from deep_generative_models.component_factory import MultiComponentFactory


class SingleOutputLayer(Module):
    model: Module

    def __init__(self, previous_layer_size: int, output_size: int, activation: Optional[Module] = None) -> None:
        super(SingleOutputLayer, self).__init__()

        if activation is None:
            self.model = Linear(previous_layer_size, output_size)
        else:
            self.model = Sequential(Linear(previous_layer_size, output_size), activation)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)


class SingleOutputLayerFactory(OutputLayerFactory):
    output_size: int
    activation: Optional[Module]

    def __init__(self, output_size: int, activation: Optional[Module] = None) -> None:
        self.output_size = output_size
        self.activation = activation

    def create(self, input_size: int) -> Module:
        return SingleOutputLayer(input_size, self.output_size, activation=self.activation)


class PartialSingleOutputLayerFactory(MultiComponentFactory):

    def mandatory_arguments(self) -> List[str]:
        return ["output_size"]

    def optional_arguments(self) -> List[str]:
        return ["activation"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        optional = {}
        if "activation" in arguments:
            activation_configuration = arguments.activation
            optional["activation"] = self.create_other(activation_configuration.factory, architecture, metadata,
                                                       activation_configuration.get("arguments", {}))
        return SingleOutputLayerFactory(arguments.output_size, **optional)

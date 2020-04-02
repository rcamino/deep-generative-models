from typing import Any, Optional, List

from torch import Tensor
from torch.nn import Module, Sequential, Linear, Tanh

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.layers.hidden_layers import HiddenLayersFactory
from deep_generative_models.layers.input_layer import InputLayer
from deep_generative_models.metadata import Metadata
from deep_generative_models.factory import MultiFactory


class Encoder(Module):

    layers: Sequential

    def __init__(self, input_layer: InputLayer, code_size: int, hidden_layers_factory: HiddenLayersFactory,
                 output_activation: Optional[Module] = None) -> None:
        super(Encoder, self).__init__()

        # input layer
        layers = [input_layer]

        # hidden layers
        hidden_layers = hidden_layers_factory.create(input_layer.get_output_size(), default_activation=Tanh())
        layers.append(hidden_layers)

        # output layer
        layers.append(Linear(hidden_layers.get_output_size(), code_size))
        if output_activation is not None:  # no default output activation
            layers.append(output_activation)

        # transform the list of layers into a sequential model
        self.layers = Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class EncoderFactory(MultiFactory):

    def mandatory_architecture_arguments(self) -> List[str]:
        return ["code_size"]

    def optional_arguments(self) -> List[str]:
        return ["hidden_layers", "output_activation"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        # create the input layer
        input_layer = self.create_input_layer(architecture, metadata, arguments)

        # create the hidden layers factory
        hidden_layers_factory = self.create_other("HiddenLayers", architecture, metadata,
                                                  arguments.get("hidden_layers", {}))

        # create the output activation
        optional = {}
        if "output_activation" in arguments:
            optional["output_activation"] = self.create_other(arguments.output_activation.factory, architecture,
                                                              metadata,
                                                              arguments.output_activation.get("arguments", {}))

        # create the encoder
        return Encoder(input_layer, architecture.arguments.code_size, hidden_layers_factory, **optional)

    def create_input_layer(self, architecture: Architecture, metadata: Metadata,
                           arguments: Configuration) -> InputLayer:
        raise NotImplementedError


class SingleInputEncoderFactory(EncoderFactory):

    def create_input_layer(self, architecture: Architecture, metadata: Metadata,
                           arguments: Configuration) -> InputLayer:
        # override the input layer size
        return self.create_other("SingleInputLayer", architecture, metadata,
                                 Configuration({"input_size": metadata.get_num_features()}))


class MultiInputEncoderFactory(EncoderFactory):

    def mandatory_arguments(self) -> List[str]:
        return ["input_layer"]

    def create_input_layer(self, architecture: Architecture, metadata: Metadata,
                           arguments: Configuration) -> InputLayer:
        return self.create_other("MultiInputLayer", architecture, metadata, arguments.input_layer)

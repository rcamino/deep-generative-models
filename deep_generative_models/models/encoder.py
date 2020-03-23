from typing import List, Any, Optional, Dict

from torch import Tensor
from torch.nn import Module, Sequential, Linear, Tanh

from deep_generative_models.configuration import Configuration
from deep_generative_models.layers.input_layer import InputLayer
from deep_generative_models.metadata import Metadata
from deep_generative_models.factory import MultiFactory


class Encoder(Module):

    layers: Sequential

    def __init__(self, input_layer: InputLayer, code_size: int, hidden_sizes: List[int] = (),
                 hidden_activation: Optional[Module] = None, output_activation: Optional[Module] = None) -> None:
        super(Encoder, self).__init__()

        # input layer
        layers = [input_layer]
        previous_layer_size = input_layer.get_output_size()

        # hidden layers
        if hidden_activation is None:  # default hidden activation
            hidden_activation = Tanh()
        for layer_size in hidden_sizes:
            layers.append(Linear(previous_layer_size, layer_size))
            layers.append(hidden_activation)
            previous_layer_size = layer_size

        # output layer
        layers.append(Linear(previous_layer_size, code_size))
        if output_activation is not None:  # no default output activation
            layers.append(output_activation)

        # transform the list of layers into a sequential model
        self.layers = Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class EncoderFactory(MultiFactory):

    def create(self, metadata: Metadata, global_configuration: Configuration, configuration: Configuration) -> Any:
        input_layer = self.create_input_layer(metadata, global_configuration, configuration)

        optional = configuration.get_all_defined(["hidden_sizes"])

        if "hidden_activation" in configuration:
            optional["hidden_activation"] = self.create_other(configuration.hidden_activation.factory,
                                                              metadata,
                                                              global_configuration,
                                                              configuration.hidden_activation.get("arguments", {}))

        if "output_activation" in configuration:
            optional["output_activation"] = self.create_other(configuration.output_activation.factory,
                                                              metadata,
                                                              global_configuration,
                                                              configuration.output_activation.get("arguments", {}))

        return Encoder(input_layer, global_configuration.code_size, **optional)

    def create_input_layer(self, metadata: Metadata, global_configuration: Configuration,
                           configuration: Configuration) -> InputLayer:
        raise NotImplementedError


class SingleInputEncoderFactory(EncoderFactory):

    def create_input_layer(self, metadata: Metadata, global_configuration: Configuration,
                           configuration: Configuration) -> InputLayer:
        # override the input layer size
        return self.create_other("SingleInputLayer", metadata, global_configuration,
                                 Configuration({"input_size": metadata.get_num_features()}))


class MultiInputEncoderFactory(EncoderFactory):

    def create_input_layer(self, metadata: Metadata, global_configuration: Configuration,
                           configuration: Configuration) -> InputLayer:
        return self.create_other("MultiInputLayer", metadata, global_configuration, configuration.input_layer)

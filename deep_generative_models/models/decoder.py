from typing import List, Any

from torch import Tensor
from torch.nn import Module, Sequential, Linear, Tanh

from deep_generative_models.configuration import Configuration
from deep_generative_models.layers.output_layer import OutputLayerFactory
from deep_generative_models.metadata import Metadata
from deep_generative_models.factory import MultiFactory


class Decoder(Module):

    hidden_layers: Sequential
    output_layer: Module

    def __init__(self, code_size: int, output_layer_factory: OutputLayerFactory, hidden_sizes: List[int] = ()) -> None:
        super(Decoder, self).__init__()

        hidden_activation = Tanh()  # TODO: check what other papers use as hidden activations

        previous_layer_size = code_size
        hidden_layers = []

        for layer_size in hidden_sizes:
            hidden_layers.append(Linear(previous_layer_size, layer_size))
            hidden_layers.append(hidden_activation)
            previous_layer_size = layer_size

        # an empty sequential module just works as the identity
        self.hidden_layers = Sequential(*hidden_layers)
        self.output_layer = output_layer_factory.create(previous_layer_size)

    def forward(self, code: Tensor) -> Tensor:
        return self.output_layer(self.hidden_layers(code))


class SingleOutputDecoderFactory(MultiFactory):

    def create(self, metadata: Metadata, global_configuration: Configuration, configuration: Configuration) -> Any:
        # override the output layer size
        output_layer_configuration = {"output_size": metadata.get_num_features()}
        # copy activation arguments only if defined
        if "output_layer" in configuration and "activation" in configuration.output_layer:
            output_layer_configuration["activation"] = configuration.output_layer.activation
        # create the output layer factory
        output_layer_factory = self.create_other("single-output layer", metadata, global_configuration,
                                                 Configuration(output_layer_configuration))
        # create the decoder
        optional = configuration.get_all_defined(["hidden_sizes"])
        return Decoder(global_configuration.code_size, output_layer_factory, **optional)


class MultiOutputDecoderFactory(MultiFactory):

    def create(self, metadata: Metadata, global_configuration: Configuration, configuration: Configuration) -> Any:
        # create the output layer factory
        output_layer_factory = self.create_other("multi-output layer", metadata, global_configuration,
                                                 configuration.output_layer)
        # create the decoder
        optional = configuration.get_all_defined(["hidden_sizes"])
        return Decoder(global_configuration.code_size, output_layer_factory, **optional)

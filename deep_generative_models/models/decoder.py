from typing import List, Any, Optional

from torch import Tensor
from torch.nn import Module, Sequential, Linear, Tanh

from deep_generative_models.configuration import Configuration
from deep_generative_models.layers.output_layer import OutputLayerFactory
from deep_generative_models.metadata import Metadata
from deep_generative_models.factory import MultiFactory


class Decoder(Module):

    hidden_layers: Sequential
    output_layer: Module

    def __init__(self, code_size: int, output_layer_factory: OutputLayerFactory, hidden_sizes: List[int] = (),
                 hidden_activation: Optional[Module] = None) -> None:
        super(Decoder, self).__init__()

        # input layer
        previous_layer_size = code_size
        hidden_layers = []

        # hidden layers
        if hidden_activation is None:  # default hidden activation
            hidden_activation = Tanh()
        for layer_size in hidden_sizes:
            hidden_layers.append(Linear(previous_layer_size, layer_size))
            hidden_layers.append(hidden_activation)
            previous_layer_size = layer_size

        # transform the list of hidden layers into a sequential model
        # an empty sequential module just works as the identity
        self.hidden_layers = Sequential(*hidden_layers)

        # output layer
        self.output_layer = output_layer_factory.create(previous_layer_size)

    def forward(self, code: Tensor) -> Tensor:
        return self.output_layer(self.hidden_layers(code))


class DecoderFactory(MultiFactory):

    def create(self, metadata: Metadata, global_configuration: Configuration, configuration: Configuration) -> Any:
        # create the output layer factory
        output_layer_factory = self.create_output_layer_factory(metadata, global_configuration, configuration)

        # create the decoder
        optional = configuration.get_all_defined(["hidden_sizes"])

        if "hidden_activation" in configuration:
            optional["hidden_activation"] = self.create_other(configuration.hidden_activation.factory,
                                                              metadata,
                                                              global_configuration,
                                                              configuration.hidden_activation.get("arguments", {}))

        return Decoder(global_configuration.code_size, output_layer_factory, **optional)

    def create_output_layer_factory(self, metadata: Metadata, global_configuration: Configuration,
                                    configuration: Configuration) -> OutputLayerFactory:
        raise NotImplementedError


class SingleOutputDecoderFactory(DecoderFactory):

    def create_output_layer_factory(self, metadata: Metadata, global_configuration: Configuration,
                                    configuration: Configuration) -> OutputLayerFactory:
        # override the output layer size
        output_layer_configuration = {"output_size": metadata.get_num_features()}
        # copy activation arguments only if defined
        if "output_layer" in configuration and "activation" in configuration.output_layer:
            output_layer_configuration["activation"] = configuration.output_layer.activation
        # create the output layer factory
        return self.create_other("SingleOutputLayer", metadata, global_configuration,
                                 Configuration(output_layer_configuration))


class MultiOutputDecoderFactory(DecoderFactory):

    def create_output_layer_factory(self, metadata: Metadata, global_configuration: Configuration,
                                    configuration: Configuration) -> OutputLayerFactory:
        # create the output layer factory
        return self.create_other("MultiOutputLayer", metadata, global_configuration, configuration.output_layer)

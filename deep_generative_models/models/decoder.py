from typing import Any

from torch import Tensor
from torch.nn import Module, Tanh, Sequential

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.layers.hidden_layers import HiddenLayersFactory
from deep_generative_models.layers.output_layer import OutputLayerFactory
from deep_generative_models.metadata import Metadata
from deep_generative_models.factory import MultiFactory


class Decoder(Module):

    layers: Sequential

    def __init__(self, code_size: int, hidden_layers_factory: HiddenLayersFactory,
                 output_layer_factory: OutputLayerFactory) -> None:
        super(Decoder, self).__init__()

        # hidden layers
        hidden_layers = hidden_layers_factory.create(code_size, default_activation=Tanh())

        # output layer
        output_layer = output_layer_factory.create(hidden_layers.get_output_size())

        # connect layers in a sequence
        self.layers = Sequential(hidden_layers, output_layer)

    def forward(self, code: Tensor) -> Tensor:
        return self.layers(code)


class DecoderFactory(MultiFactory):

    def create(self, architecture: Architecture, metadata: Metadata, global_configuration: Configuration,
               configuration: Configuration) -> Any:
        # create the output layer factory
        output_layer_factory = self.create_output_layer_factory(architecture, metadata, global_configuration,
                                                                configuration)

        # create the hidden layers factory
        hidden_layers_factory = self.create_other("HiddenLayers",
                                                  architecture,
                                                  metadata,
                                                  global_configuration,
                                                  configuration.get("hidden_layers", {}))

        # create the decoder
        return Decoder(global_configuration.code_size, hidden_layers_factory, output_layer_factory)

    def create_output_layer_factory(self, architecture: Architecture, metadata: Metadata,
                                    global_configuration: Configuration,
                                    configuration: Configuration) -> OutputLayerFactory:
        raise NotImplementedError


class SingleOutputDecoderFactory(DecoderFactory):

    def create_output_layer_factory(self, architecture: Architecture, metadata: Metadata,
                                    global_configuration: Configuration,
                                    configuration: Configuration) -> OutputLayerFactory:
        # override the output layer size
        output_layer_configuration = {"output_size": metadata.get_num_features()}
        # copy activation arguments only if defined
        if "output_layer" in configuration and "activation" in configuration.output_layer:
            output_layer_configuration["activation"] = configuration.output_layer.activation
        # create the output layer factory
        return self.create_other("SingleOutputLayer", architecture, metadata, global_configuration,
                                 Configuration(output_layer_configuration))


class MultiOutputDecoderFactory(DecoderFactory):

    def create_output_layer_factory(self, architecture: Architecture, metadata: Metadata,
                                    global_configuration: Configuration,
                                    configuration: Configuration) -> OutputLayerFactory:
        # create the output layer factory
        return self.create_other("MultiOutputLayer", architecture, metadata, global_configuration,
                                 configuration.output_layer)

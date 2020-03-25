from typing import Any

from torch import Tensor
from torch.nn import Module, ReLU, Sequential

from deep_generative_models.configuration import Configuration
from deep_generative_models.factory import MultiFactory
from deep_generative_models.layers.hidden_layers import HiddenLayersFactory
from deep_generative_models.layers.output_layer import OutputLayerFactory
from deep_generative_models.metadata import Metadata


class Generator(Module):

    layers: Sequential

    def __init__(self, noise_size: int, hidden_layers_factory: HiddenLayersFactory,
                 output_layer_factory: OutputLayerFactory) -> None:
        super(Generator, self).__init__()

        # hidden layers
        hidden_layers = hidden_layers_factory.create(noise_size, default_activation=ReLU())

        # output layer
        output_layer = output_layer_factory.create(hidden_layers.get_output_size())

        # connect layers in a sequence
        self.layers = Sequential(hidden_layers, output_layer)

    def forward(self, noise: Tensor) -> Tensor:
        return self.layers(noise)


class GeneratorFactory(MultiFactory):

    def create(self, metadata: Metadata, global_configuration: Configuration, configuration: Configuration) -> Any:
        # create the output layer factory
        output_layer_factory = self.create_output_layer_factory(metadata, global_configuration, configuration)

        # create the hidden layers factory
        hidden_layers_factory = self.create_other("HiddenLayers",
                                                  metadata,
                                                  global_configuration,
                                                  configuration.get("hidden_layers", {}))

        # create the generator
        return Generator(global_configuration.noise_size, hidden_layers_factory, output_layer_factory)

    def create_output_layer_factory(self, metadata: Metadata, global_configuration: Configuration,
                                    configuration: Configuration) -> OutputLayerFactory:
        raise NotImplementedError


class SingleOutputGeneratorFactory(GeneratorFactory):

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


class MultiOutputGeneratorFactory(GeneratorFactory):

    def create_output_layer_factory(self, metadata: Metadata, global_configuration: Configuration,
                                    configuration: Configuration) -> OutputLayerFactory:
        # create the output layer factory
        return self.create_other("MultiOutputLayer", metadata, global_configuration, configuration.output_layer)

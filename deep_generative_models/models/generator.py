from typing import List, Any

from torch import Tensor
from torch.nn import Module, Sequential, Linear, ReLU, BatchNorm1d

from deep_generative_models.configuration import Configuration
from deep_generative_models.factory import MultiFactory
from deep_generative_models.layers.output_layer import OutputLayerFactory
from deep_generative_models.metadata import Metadata


class Generator(Module):

    hidden_layers: Sequential
    output_layer: Module

    def __init__(self, noise_size: int, output_layer_factory: OutputLayerFactory, hidden_sizes: List[int] = (),
                 bn_decay: float = 0):
        super(Generator, self).__init__()

        # input layer
        previous_layer_size = noise_size
        hidden_layers = []

        # hidden layers
        hidden_activation = ReLU()  # TODO: parametrize?
        for layer_number, layer_size in enumerate(hidden_sizes):
            hidden_layers.append(Linear(previous_layer_size, layer_size))
            if layer_number > 0 and bn_decay > 0:
                hidden_layers.append(BatchNorm1d(layer_size, momentum=(1 - bn_decay)))
            hidden_layers.append(hidden_activation)
            previous_layer_size = layer_size

        # transform the list of hidden layers into a sequential model
        # an empty sequential module just works as the identity
        self.hidden_layers = Sequential(*hidden_layers)

        # output layer
        self.output_layer = output_layer_factory.create(previous_layer_size)

    def forward(self, noise: Tensor) -> Tensor:
        return self.output_layer(self.hidden_layers(noise))


class SingleOutputGeneratorFactory(MultiFactory):

    def create(self, metadata: Metadata, global_configuration: Configuration, configuration: Configuration) -> Any:
        # override the output layer size
        output_layer_configuration = {"output_size": metadata.get_num_features()}
        # copy activation arguments only if defined
        if "output_layer" in configuration and "activation" in configuration.output_layer:
            output_layer_configuration["activation"] = configuration.output_layer.activation
        # create the output layer factory
        output_layer_factory = self.create_other("SingleOutputLayer", metadata, global_configuration,
                                                 Configuration(output_layer_configuration))
        # create the generator
        optional = configuration.get_all_defined(["hidden_sizes", "bn_decay"])
        return Generator(global_configuration.noise_size, output_layer_factory, **optional)


class MultiOutputGeneratorFactory(MultiFactory):

    def create(self, metadata: Metadata, global_configuration: Configuration, configuration: Configuration) -> Any:
        # create the output layer factory
        output_layer_factory = self.create_other("MultiOutputLayer", metadata, global_configuration,
                                                 configuration.output_layer)
        # create the generator
        optional = configuration.get_all_defined(["hidden_sizes", "bn_decay"])
        return Generator(global_configuration.noise_size, output_layer_factory, **optional)

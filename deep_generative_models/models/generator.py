from typing import Any, List, Optional

from torch import Tensor
from torch.nn import Module, ReLU, Sequential

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.component_factory import MultiComponentFactory
from deep_generative_models.layers.conditional_layer import ConditionalLayer
from deep_generative_models.layers.hidden_layers import HiddenLayersFactory
from deep_generative_models.layers.input_layer import InputLayer
from deep_generative_models.layers.output_layer import OutputLayerFactory
from deep_generative_models.metadata import Metadata


class Generator(Module):

    input_layer: InputLayer
    layers: Sequential

    def __init__(self, input_layer: InputLayer, hidden_layers_factory: HiddenLayersFactory,
                 output_layer_factory: OutputLayerFactory) -> None:
        super(Generator, self).__init__()

        # input layer
        self.input_layer = input_layer

        # hidden layers
        hidden_layers = hidden_layers_factory.create(input_layer.get_output_size(), default_activation=ReLU())

        # output layer
        output_layer = output_layer_factory.create(hidden_layers.get_output_size())

        # connect layers in a sequence
        self.layers = Sequential(hidden_layers, output_layer)

    def forward(self, noise: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        return self.layers(self.input_layer(noise, condition=condition))


class GeneratorFactory(MultiComponentFactory):

    def mandatory_architecture_arguments(self) -> List[str]:
        return ["noise_size"]

    def optional_arguments(self) -> List[str]:
        return ["hidden_layers"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        # create input layer
        input_layer = self.create_other("SingleInputLayer", architecture, metadata,
                                        Configuration({"input_size": architecture.arguments.noise_size}))

        # conditional
        if "conditional" in architecture.arguments:
            # wrap the input layer with a conditional layer
            input_layer = ConditionalLayer(input_layer, metadata, **architecture.arguments.conditional)

        # create the output layer factory
        output_layer_factory = self.create_output_layer_factory(architecture, metadata, arguments)

        # create the hidden layers factory
        hidden_layers_factory = self.create_other("HiddenLayers", architecture, metadata,
                                                  arguments.get("hidden_layers", {}))

        # create the generator
        return Generator(input_layer, hidden_layers_factory, output_layer_factory)

    def create_output_layer_factory(self, architecture: Architecture, metadata: Metadata,
                                    arguments: Configuration) -> OutputLayerFactory:
        raise NotImplementedError


class SingleOutputGeneratorFactory(GeneratorFactory):

    def optional_arguments(self) -> List[str]:
        return ["output_layer"] + super(SingleOutputGeneratorFactory, self).optional_arguments()

    def create_output_layer_factory(self, architecture: Architecture, metadata: Metadata,
                                    arguments: Configuration) -> OutputLayerFactory:
        # override the output layer size
        output_layer_configuration = {"output_size": metadata.get_num_features()}
        # copy activation arguments only if defined
        if "output_layer" in arguments and "activation" in arguments.output_layer:
            output_layer_configuration["activation"] = arguments.output_layer.activation
        # create the output layer factory
        return self.create_other("SingleOutputLayer", architecture, metadata, Configuration(output_layer_configuration))


class MultiOutputGeneratorFactory(GeneratorFactory):

    def mandatory_arguments(self) -> List[str]:
        return ["output_layer"]

    def create_output_layer_factory(self, architecture: Architecture, metadata: Metadata,
                                    arguments: Configuration) -> OutputLayerFactory:
        # create the output layer factory
        return self.create_other("MultiOutputLayer", architecture, metadata, arguments.output_layer)

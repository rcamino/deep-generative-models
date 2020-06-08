from typing import Any, List

from torch.nn import Tanh

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.layers.gain_input_layer import GAINInputLayer
from deep_generative_models.layers.input_layer import InputLayer
from deep_generative_models.layers.output_layer import OutputLayerFactory
from deep_generative_models.metadata import Metadata
from deep_generative_models.component_factory import MultiComponentFactory
from deep_generative_models.models.feed_forward import FeedForward


class GAINGeneratorFactory(MultiComponentFactory):

    def optional_arguments(self) -> List[str]:
        return ["hidden_layers"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        # create the input layer
        input_layer = self.create_input_layer(architecture, metadata, arguments)
        # wrap the input layer with the special gain input layer (to receive the mask)
        input_layer = GAINInputLayer(input_layer, metadata.get_num_features())

        # create the hidden layers factory
        hidden_layers_factory = self.create_other("HiddenLayers", architecture, metadata,
                                                  arguments.get("hidden_layers", {}))

        # create the output layer factory
        output_layer_factory = self.create_output_layer_factory(architecture, metadata, arguments)

        # create the encoder
        return FeedForward(input_layer, hidden_layers_factory, output_layer_factory, default_hidden_activation=Tanh())

    def create_input_layer(self, architecture: Architecture, metadata: Metadata,
                           arguments: Configuration) -> InputLayer:
        raise NotImplementedError

    def create_output_layer_factory(self, architecture: Architecture, metadata: Metadata,
                                    arguments: Configuration) -> OutputLayerFactory:
        raise NotImplementedError


class SingleVariableGAINGeneratorFactory(GAINGeneratorFactory):

    def optional_arguments(self) -> List[str]:
        return ["output_layer"] + super(SingleVariableGAINGeneratorFactory, self).optional_arguments()

    def create_input_layer(self, architecture: Architecture, metadata: Metadata,
                           arguments: Configuration) -> InputLayer:
        # override the input layer size
        return self.create_other("SingleInputLayer", architecture, metadata,
                                 Configuration({"input_size": metadata.get_num_features()}))

    def create_output_layer_factory(self, architecture: Architecture, metadata: Metadata,
                                    arguments: Configuration) -> OutputLayerFactory:
        # override the output layer size
        output_layer_configuration = {"output_size": metadata.get_num_features()}
        # copy activation arguments only if defined
        if "output_layer" in arguments and "activation" in arguments.output_layer:
            output_layer_configuration["activation"] = arguments.output_layer.activation
        # create the output layer factory
        return self.create_other("SingleOutputLayer", architecture, metadata, Configuration(output_layer_configuration))


class MultiVariableGAINGeneratorFactory(GAINGeneratorFactory):

    def mandatory_arguments(self) -> List[str]:
        return ["input_layer", "output_layer"]

    def create_input_layer(self, architecture: Architecture, metadata: Metadata,
                           arguments: Configuration) -> InputLayer:
        return self.create_other("MultiInputLayer", architecture, metadata, arguments.input_layer)

    def create_output_layer_factory(self, architecture: Architecture, metadata: Metadata,
                                    arguments: Configuration) -> OutputLayerFactory:
        return self.create_other("MultiOutputLayer", architecture, metadata, arguments.output_layer)

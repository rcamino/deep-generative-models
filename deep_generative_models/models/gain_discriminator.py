from typing import Any, List

from torch.nn import Tanh, Sigmoid

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.layers.gain_input_layer import GAINInputLayer
from deep_generative_models.layers.input_layer import InputLayer
from deep_generative_models.layers.single_output_layer import SingleOutputLayerFactory
from deep_generative_models.metadata import Metadata
from deep_generative_models.component_factory import MultiComponentFactory
from deep_generative_models.models.feed_forward import FeedForward


class GAINDiscriminatorFactory(MultiComponentFactory):

    def optional_arguments(self) -> List[str]:
        return ["hidden_layers"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        # create the input layer
        input_layer = self.create_input_layer(architecture, metadata, arguments)
        # wrap the input layer with the special gain input layer (to receive the mask)
        input_layer = GAINInputLayer(input_layer)

        # create the hidden layers factory
        hidden_layers_factory = self.create_other("HiddenLayers", architecture, metadata,
                                                  arguments.get("hidden_layers", {}))

        # create the output layer factory
        # this is different from a normal discriminator
        # because the output has the size of the input
        # it predicts if each feature is real or fake
        output_layer_factory = SingleOutputLayerFactory(metadata.get_num_features(), activation=Sigmoid())

        # create the encoder
        return FeedForward(input_layer, hidden_layers_factory, output_layer_factory, default_hidden_activation=Tanh())

    def create_input_layer(self, architecture: Architecture, metadata: Metadata,
                           arguments: Configuration) -> InputLayer:
        raise NotImplementedError


class SingleInputGAINDiscriminatorFactory(GAINDiscriminatorFactory):

    def create_input_layer(self, architecture: Architecture, metadata: Metadata,
                           arguments: Configuration) -> InputLayer:
        # override the input layer size
        return self.create_other("SingleInputLayer", architecture, metadata,
                                 Configuration({"input_size": metadata.get_num_features()}))


class MultiInputGAINDiscriminatorFactory(GAINDiscriminatorFactory):

    def mandatory_arguments(self) -> List[str]:
        return ["input_layer"]

    def create_input_layer(self, architecture: Architecture, metadata: Metadata,
                           arguments: Configuration) -> InputLayer:
        return self.create_other("MultiInputLayer", architecture, metadata, arguments.input_layer)

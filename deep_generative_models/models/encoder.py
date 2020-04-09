from typing import Any, List

from torch.nn import Tanh

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.layers.conditional_layer import ConditionalLayer
from deep_generative_models.layers.input_layer import InputLayer
from deep_generative_models.layers.single_output_layer import SingleOutputLayerFactory
from deep_generative_models.metadata import Metadata
from deep_generative_models.component_factory import MultiComponentFactory
from deep_generative_models.models.feed_forward import FeedForward


class EncoderFactory(MultiComponentFactory):

    def mandatory_architecture_arguments(self) -> List[str]:
        return ["code_size"]

    def optional_arguments(self) -> List[str]:
        return ["hidden_layers", "output_activation"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        # create the input layer
        input_layer = self.create_input_layer(architecture, metadata, arguments)

        # conditional
        if "conditional" in architecture.arguments:
            # wrap the input layer with a conditional layer
            input_layer = ConditionalLayer(input_layer, metadata, **architecture.arguments.conditional)

        # create the hidden layers factory
        hidden_layers_factory = self.create_other("HiddenLayers", architecture, metadata,
                                                  arguments.get("hidden_layers", {}))

        # create the output activation
        if "output_activation" in arguments:
            output_activation = self.create_other(arguments.output_activation.factory, architecture,
                                                  metadata,
                                                  arguments.output_activation.get("arguments", {}))
        else:
            output_activation = None

        # create the output layer factory
        output_layer_factory = SingleOutputLayerFactory(architecture.arguments.code_size, activation=output_activation)

        # create the encoder
        return FeedForward(input_layer, hidden_layers_factory, output_layer_factory, default_hidden_activation=Tanh())

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

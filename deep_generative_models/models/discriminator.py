from typing import Any, Dict, List

from torch.nn import LeakyReLU, Sigmoid, Sequential

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.component_factory import MultiComponentFactory, ComponentFactory
from deep_generative_models.layers.conditional_layer import ConditionalLayer
from deep_generative_models.layers.single_output_layer import SingleOutputLayerFactory
from deep_generative_models.layers.view import View
from deep_generative_models.metadata import Metadata
from deep_generative_models.models.feed_forward import FeedForward


class DiscriminatorFactory(MultiComponentFactory):
    critic: bool
    code: bool

    def __init__(self, factory_by_name: Dict[str, ComponentFactory], critic: bool = False, code: bool = False) -> None:
        super(DiscriminatorFactory, self).__init__(factory_by_name)
        self.critic = critic
        self.code = code

    def mandatory_architecture_arguments(self) -> List[str]:
        if self.code:
            return ["code_size"]
        else:
            return []

    def optional_arguments(self) -> List[str]:
        return ["hidden_layers"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        # create input layer
        input_layer_configuration = {}
        if self.code:
            input_layer_configuration["input_size"] = architecture.arguments.code_size
        else:
            input_layer_configuration["input_size"] = metadata.get_num_features()

        input_layer = self.create_other("SingleInputLayer", architecture, metadata,
                                        Configuration(input_layer_configuration))

        # conditional
        if "conditional" in architecture.arguments:
            # wrap the input layer with a conditional layer
            input_layer = ConditionalLayer(input_layer, metadata, **architecture.arguments.conditional)

        # create the hidden layers factory
        hidden_layers_factory = self.create_other("HiddenLayers", architecture, metadata,
                                                  arguments.get("hidden_layers", {}))

        # create the output activation
        if self.critic:
            output_activation = View(-1)
        else:
            output_activation = Sequential(Sigmoid(), View(-1))

        # create the output layer factory
        output_layer_factory = SingleOutputLayerFactory(1, activation=output_activation)

        # create the discriminator
        return FeedForward(input_layer, hidden_layers_factory, output_layer_factory,
                           default_hidden_activation=LeakyReLU(0.2))

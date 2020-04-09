from typing import Any, Dict, List, Optional

from torch import Tensor
from torch.nn import Module, Sequential, Linear, LeakyReLU, Sigmoid

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.component_factory import MultiComponentFactory, ComponentFactory
from deep_generative_models.layers.conditional_layer import ConditionalLayer
from deep_generative_models.layers.hidden_layers import HiddenLayersFactory
from deep_generative_models.layers.input_layer import InputLayer
from deep_generative_models.metadata import Metadata


class Discriminator(Module):

    input_layer: InputLayer
    layers: Sequential

    def __init__(self, input_layer: InputLayer, hidden_layers_factory: HiddenLayersFactory,
                 critic: bool = False) -> None:
        super(Discriminator, self).__init__()

        # input layer
        self.input_layer = input_layer

        # hidden layers
        hidden_layers = hidden_layers_factory.create(input_layer.get_output_size(), default_activation=LeakyReLU(0.2))

        # concat hidden layers with output layer
        layers = [
            hidden_layers,
            Linear(hidden_layers.get_output_size(), 1)
        ]

        # add the activation
        # unless it is a critic, which has linear output
        if not critic:
            layers.append(Sigmoid())

        # transform the list of layers into a sequential model
        self.layers = Sequential(*layers)

    def forward(self, inputs: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        return self.layers(self.input_layer(inputs, condition=condition)).view(-1)


class DiscriminatorFactory(MultiComponentFactory):
    critic: bool

    def __init__(self, factory_by_name: Dict[str, ComponentFactory], critic: bool = False) -> None:
        super(DiscriminatorFactory, self).__init__(factory_by_name)
        self.critic = critic

    def optional_arguments(self) -> List[str]:
        return ["hidden_layers"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        # create input layer
        input_layer = self.create_other("SingleInputLayer", architecture, metadata,
                                        Configuration({"input_size": metadata.get_num_features()}))

        # conditional
        if "conditional" in architecture.arguments:
            # wrap the input layer with a conditional layer
            input_layer = ConditionalLayer(input_layer, metadata, **architecture.arguments.conditional)

        # create the hidden layers factory
        hidden_layers_factory = self.create_other("HiddenLayers", architecture, metadata,
                                                  arguments.get("hidden_layers", {}))

        # create the discriminator
        return Discriminator(input_layer, hidden_layers_factory, critic=self.critic)

from typing import Any, Dict, List

from torch import Tensor
from torch.nn import Module, Sequential, Linear, LeakyReLU, Sigmoid

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.component_factory import MultiComponentFactory, ComponentFactory
from deep_generative_models.layers.hidden_layers import HiddenLayersFactory
from deep_generative_models.metadata import Metadata


class Discriminator(Module):

    layers: Sequential

    def __init__(self, input_size: int, hidden_layers_factory: HiddenLayersFactory, critic: bool = False) -> None:
        super(Discriminator, self).__init__()

        # hidden layers
        hidden_layers = hidden_layers_factory.create(input_size, default_activation=LeakyReLU(0.2))

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

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs).view(-1)


class DiscriminatorFactory(MultiComponentFactory):
    critic: bool

    def __init__(self, factory_by_name: Dict[str, ComponentFactory], critic: bool = False) -> None:
        super(DiscriminatorFactory, self).__init__(factory_by_name)
        self.critic = critic

    def optional_arguments(self) -> List[str]:
        return ["hidden_layers"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        # create the hidden layers factory
        hidden_layers_factory = self.create_other("HiddenLayers", architecture, metadata,
                                                  arguments.get("hidden_layers", {}))

        # create the discriminator
        return Discriminator(metadata.get_num_features(), hidden_layers_factory, critic=self.critic)

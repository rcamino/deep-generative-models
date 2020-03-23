from typing import List, Any, Optional

from torch import Tensor
from torch.nn import Module, Sequential, Linear, LeakyReLU, Sigmoid, BatchNorm1d

from deep_generative_models.configuration import Configuration
from deep_generative_models.factory import MultiFactory
from deep_generative_models.metadata import Metadata


class Discriminator(Module):

    layers: Sequential

    def __init__(self, input_size: int, hidden_sizes: List[int] = (), bn_decay: float = 0,
                 hidden_activation: Optional[Module] = None, critic: bool = False) -> None:
        super(Discriminator, self).__init__()

        # input layer
        previous_layer_size = input_size
        layers = []

        # hidden layers
        if hidden_activation is None:  # default hidden activation
            hidden_activation = LeakyReLU(0.2)
        for layer_number, layer_size in enumerate(hidden_sizes):
            layers.append(Linear(previous_layer_size, layer_size))
            if layer_number > 0 and bn_decay > 0:
                layers.append(BatchNorm1d(layer_size, momentum=(1 - bn_decay)))
            layers.append(hidden_activation)
            previous_layer_size = layer_size

        # output layer
        layers.append(Linear(previous_layer_size, 1))

        # the critic has a linear output
        if not critic:
            layers.append(Sigmoid())

        # transform the list of layers into a sequential model
        self.layers = Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs).view(-1)


class DiscriminatorFactory(MultiFactory):

    def create(self, metadata: Metadata, global_configuration: Configuration, configuration: Configuration) -> Any:
        optional = configuration.get_all_defined(["hidden_sizes", "bn_decay", "critic"])

        if "hidden_activation" in configuration:
            optional["hidden_activation"] = self.create_other(configuration.hidden_activation.factory,
                                                              metadata,
                                                              global_configuration,
                                                              configuration.hidden_activation.get("arguments", {}))

        return Discriminator(metadata.get_num_features(), **optional)

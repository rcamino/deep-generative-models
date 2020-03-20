from typing import List

from torch import Tensor
from torch.nn import Module, Sequential, Linear, LeakyReLU, Sigmoid, BatchNorm1d


class Discriminator(Module):

    layers: Sequential

    def __init__(self, input_size: int, hidden_sizes: List[int] = (), bn_decay: float = 0,
                 critic: bool = False) -> None:
        super(Discriminator, self).__init__()

        hidden_activation = LeakyReLU(0.2)

        previous_layer_size = input_size
        layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):
            layers.append(Linear(previous_layer_size, layer_size))
            if layer_number > 0 and bn_decay > 0:
                layers.append(BatchNorm1d(layer_size, momentum=(1 - bn_decay)))
            layers.append(hidden_activation)
            previous_layer_size = layer_size

        layers.append(Linear(previous_layer_size, 1))

        # the critic has a linear output
        if not critic:
            layers.append(Sigmoid())

        self.layers = Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs).view(-1)

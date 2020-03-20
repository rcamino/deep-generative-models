from typing import List

from torch import Tensor
from torch.nn import Module, Sequential, Linear, ReLU, BatchNorm1d

from deep_generative_models.layers.output_layer import OutputLayerFactory


class Generator(Module):

    def __init__(self, noise_size: int, output_layer_factory: OutputLayerFactory, hidden_sizes: List[int] = (),
                 bn_decay: float = 0):
        super(Generator, self).__init__()

        hidden_activation = ReLU()

        previous_layer_size = noise_size
        hidden_layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):
            hidden_layers.append(Linear(previous_layer_size, layer_size))
            if layer_number > 0 and bn_decay > 0:
                hidden_layers.append(BatchNorm1d(layer_size, momentum=(1 - bn_decay)))
            hidden_layers.append(hidden_activation)
            previous_layer_size = layer_size

        # an empty sequential module just works as the identity
        self.hidden_layers = Sequential(*hidden_layers)
        self.output_layer = output_layer_factory.create(previous_layer_size)

    def forward(self, noise: Tensor) -> Tensor:
        return self.output_layer(self.hidden_layers(noise))

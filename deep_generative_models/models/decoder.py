from typing import List

from torch import Tensor
from torch.nn import Module, Sequential, Linear, Tanh

from deep_generative_models.layers.output_layer import OutputLayerFactory


class Decoder(Module):

    hidden_layers: Sequential
    output_layer: Module

    def __init__(self, code_size: int, output_layer_factory: OutputLayerFactory, hidden_sizes: List[int] = ()) -> None:
        super(Decoder, self).__init__()

        hidden_activation = Tanh()  # TODO: check what other papers use as hidden activations

        previous_layer_size = code_size
        hidden_layers = []

        for layer_size in hidden_sizes:
            hidden_layers.append(Linear(previous_layer_size, layer_size))
            hidden_layers.append(hidden_activation)
            previous_layer_size = layer_size

        # an empty sequential module just works as the identity
        self.hidden_layers = Sequential(*hidden_layers)
        self.output_layer = output_layer_factory.create(previous_layer_size)

    def forward(self, code: Tensor) -> Tensor:
        return self.output_layer(self.hidden_layers(code))

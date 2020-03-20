from typing import List

from torch import Tensor
from torch.nn import Module, Sequential, Linear, Tanh

from deep_generative_models.layers.input_layer import InputLayer


class Encoder(Module):

    layers: Sequential

    def __init__(self, input_layer: InputLayer, code_size: int, hidden_sizes: List[int] = ()) -> None:
        super(Encoder, self).__init__()

        layers = [input_layer]
        previous_layer_size = input_layer.get_output_size()

        layer_sizes = list(hidden_sizes) + [code_size]
        activation = Tanh()  # TODO: check what other papers use as hidden and output activations

        for layer_size in layer_sizes:
            layers.append(Linear(previous_layer_size, layer_size))
            layers.append(activation)
            previous_layer_size = layer_size

        self.layers = Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)

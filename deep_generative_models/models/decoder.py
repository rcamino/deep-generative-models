from typing import Optional, List

from torch import Tensor
from torch.nn import Module, Sequential, Linear, ReLU

from deep_generative_models.layers.multi_output import MultiOutputLayer
from deep_generative_models.layers.single_output import SingleOutputLayer

from deep_generative_models.metadata import Metadata


class Decoder(Module):

    hidden_layers: Optional[Sequential]
    output_layer: Module

    def __init__(self, code_size: int, output_size: int, hidden_sizes: List[int] = (),
                 metadata: Optional[Metadata] = None, temperature: Optional[float] = None) -> None:
        super(Decoder, self).__init__()

        hidden_activation = ReLU()

        previous_layer_size = code_size
        hidden_layers = []

        for layer_size in hidden_sizes:
            hidden_layers.append(Linear(previous_layer_size, layer_size))
            hidden_layers.append(hidden_activation)
            previous_layer_size = layer_size

        if len(hidden_layers) > 0:
            self.hidden_layers = Sequential(*hidden_layers)
        else:
            self.hidden_layers = None

        if metadata is None:
            self.output_layer = SingleOutputLayer(previous_layer_size, output_size)
        else:
            self.output_layer = MultiOutputLayer(previous_layer_size, metadata, temperature=temperature)

    def forward(self, code: Tensor) -> Tensor:
        if self.hidden_layers is None:
            hidden = code
        else:
            hidden = self.hidden_layers(code)

        return self.output_layer(hidden)

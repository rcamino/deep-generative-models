from typing import List, Any

from torch import Tensor
from torch.nn import Module, Sequential, Linear, Tanh

from deep_generative_models.configuration import Configuration
from deep_generative_models.layers.input_layer import InputLayer
from deep_generative_models.metadata import Metadata
from deep_generative_models.factory import MultiFactory


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


class SingleInputEncoderFactory(MultiFactory):

    def create(self, metadata: Metadata, global_configuration: Configuration, configuration: Configuration) -> Any:
        # override the input layer size
        input_layer = self.create_other("single-input layer", metadata, global_configuration,
                                        Configuration({"input_size": metadata.get_num_features()}))
        optional = configuration.get_all_defined(["hidden_sizes"])
        return Encoder(input_layer, global_configuration.code_size, **optional)


class MultiInputEncoderFactory(MultiFactory):

    def create(self, metadata: Metadata, global_configuration: Configuration, configuration: Configuration) -> Any:
        input_layer = self.create_other("multi-input layer", metadata, global_configuration,
                                        configuration.input_layer)
        optional = configuration.get_all_defined(["hidden_sizes"])
        return Encoder(input_layer, global_configuration.code_size, **optional)

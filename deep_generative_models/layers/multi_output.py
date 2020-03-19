import torch

from typing import List, Optional

from torch import Tensor
from torch.nn import Module, Linear, Sequential
from torch.nn.functional import gumbel_softmax, softmax

from torch.distributions.one_hot_categorical import OneHotCategorical

from deep_generative_models.metadata import Metadata


class OutputBinaryVariableActivation(Module):

    def __init__(self) -> None:
        super(OutputBinaryVariableActivation, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.sigmoid(inputs)


class OutputCategoricalVariableActivation(Module):
    temperature: float

    def __init__(self, temperature: float) -> None:
        super(OutputCategoricalVariableActivation, self).__init__()
        self.temperature = temperature

    def forward(self, inputs: Tensor) -> Tensor:
        # gumbel-softmax (training and evaluation)
        if self.temperature is not None:
            return gumbel_softmax(inputs, hard=not self.training, tau=self.temperature)
        # softmax training
        elif self.training:
            return softmax(inputs, dim=1)
        # softmax evaluation
        else:
            return OneHotCategorical(logits=inputs).sample()


class MultiOutputLayer(Module):
    layers: List[Module]

    def __init__(self, input_size: int, metadata: Metadata, temperature: float = None) -> None:
        super(MultiOutputLayer, self).__init__()

        self.layers = []

        # accumulate binary or numerical variables into "blocks"
        current_block_type = None
        current_block_size = 0
        current_block_index = 1

        for variable_metadata in metadata.get_by_variable():
            # first check if a block needs to be created
            if current_block_size > 0 and variable_metadata.get_type() != current_block_type:
                # create the block
                self._add_block(current_block_type, current_block_size, current_block_index, input_size)
                # empty the accumulated data
                current_block_type = None
                current_block_size = 0
                # move the block index
                current_block_index += 1

            # if it is a binary or numerical variable
            if variable_metadata.is_binary() or variable_metadata.is_numerical():
                assert variable_metadata.get_size() == 1
                current_block_type = variable_metadata.get_type()
                current_block_size += 1

            # if it is a categorical variable
            elif variable_metadata.is_categorical():
                # create the categorical layer
                self._add_layer(variable_metadata.get_name(),
                                input_size, variable_metadata.get_size(),
                                OutputCategoricalVariableActivation(temperature))

            # if it is another type
            else:
                raise Exception("Unexpected variable type '{}' for variable '{}'.".format(
                    variable_metadata.get_type(), variable_metadata.get_name()))

        # if there is still accumulated data for a block
        if current_block_size > 0:
            # create the last block
            self._add_block(current_block_type, current_block_size, current_block_index, input_size)

    def _add_layer(self, name: str, input_size: int, output_size: int, activation: Optional[Module] = None) -> None:
        if activation is None:
            layer = Linear(input_size, output_size)
        else:
            layer = Sequential(Linear(input_size, output_size), activation)
        self.layers.append(layer)
        self.add_module(name, layer)

    def _add_block(self, block_type: str, block_size: int, block_index: int, input_size: int) -> None:
        name = "block_{:d}_{}".format(block_index, block_type)

        if block_type == "binary":
            activation = OutputBinaryVariableActivation()
        elif block_type == "numerical":
            activation = None
        else:
            raise Exception("Unexpected block type '{}'.".format(block_type))

        self._add_layer(name, input_size, block_size, activation)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = [layer(inputs) for layer in self.layers]
        return torch.cat(outputs, dim=1)

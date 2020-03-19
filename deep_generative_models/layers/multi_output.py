import torch

from typing import Optional

from torch import Tensor
from torch.nn import Module, Linear, Sequential, ModuleList
from torch.nn.functional import gumbel_softmax, softmax

from torch.distributions.one_hot_categorical import OneHotCategorical

from deep_generative_models.metadata import Metadata, VariableMetadata


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


class BlockBuilder:
    variable_metadata: VariableMetadata
    activation: Optional[Module]
    size: int

    def __init__(self, variable_metadata: VariableMetadata) -> None:
        if not (variable_metadata.is_binary() or variable_metadata.is_numerical()):
            raise Exception("Unexpected variable type '{}' for block.".format(variable_metadata.get_type()))

        self.variable_metadata = variable_metadata
        self.size = 0
        self.add(variable_metadata)

    def matches_type(self, variable_metadata: VariableMetadata) -> bool:
        return self.variable_metadata.get_type() == variable_metadata.get_type()

    def add(self, variable_metadata: VariableMetadata) -> None:
        assert self.matches_type(variable_metadata)
        self.size += 1

    def build(self, input_size: int) -> Module:
        if self.variable_metadata.is_binary():
            return Sequential(Linear(input_size, self.variable_metadata.get_size()), OutputBinaryVariableActivation())
        elif self.variable_metadata.is_numerical():
            return Linear(input_size, self.variable_metadata.get_size())


class MultiOutputLayer(Module):
    layers: ModuleList

    def __init__(self, input_size: int, metadata: Metadata, temperature: Optional[float] = None) -> None:
        super(MultiOutputLayer, self).__init__()

        self.layers = ModuleList()

        # accumulate binary or numerical variables into "blocks"
        current_block = None

        for variable_metadata in metadata.get_by_variable():
            # first check if a block needs to be created
            if current_block is not None and not current_block.matches_type(variable_metadata):
                # create the block
                self.layers.append(current_block.build(input_size))
                # empty the block
                current_block = None

            # if it is a binary or numerical variable
            if variable_metadata.is_binary() or variable_metadata.is_numerical():
                # create a block
                if current_block is None:
                    current_block = BlockBuilder(variable_metadata)
                # or add to the existing block
                else:
                    current_block.add(variable_metadata)

            # if it is a categorical variable
            elif variable_metadata.is_categorical():
                # create the categorical layer
                self.layers.append(Sequential(
                    Linear(input_size, variable_metadata.get_size()),
                    OutputCategoricalVariableActivation(temperature)))

            # if it is another type
            else:
                raise Exception("Unexpected variable type '{}' for variable '{}'.".format(
                    variable_metadata.get_type(), variable_metadata.get_name()))

        # if there is still accumulated data for a block
        if current_block is not None:
            # create the last block
            self.layers.append(current_block.build(input_size))

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.cat([layer(inputs) for layer in self.layers], dim=1)

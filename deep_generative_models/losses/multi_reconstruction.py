from typing import Any

import torch

from torch import Tensor
from torch.nn import Module
from torch.nn.functional import cross_entropy, binary_cross_entropy, mse_loss

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.factory import Factory
from deep_generative_models.metadata import Metadata, VariableMetadata


class BlockLossBuilder:
    variable_metadata: VariableMetadata
    size: int

    def __init__(self, variable_metadata: VariableMetadata) -> None:
        if not (variable_metadata.is_binary() or variable_metadata.is_numerical()):
            raise Exception("Unexpected variable type '{}' for block.".format(variable_metadata.get_type()))

        self.variable_metadata = variable_metadata
        self.size = 0
        self.add(variable_metadata)

    def get_size(self) -> int:
        return self.size

    def matches_type(self, variable_metadata: VariableMetadata) -> bool:
        return self.variable_metadata.get_type() == variable_metadata.get_type()

    def add(self, variable_metadata: VariableMetadata) -> None:
        assert self.matches_type(variable_metadata)
        self.size += 1

    def build(self, reconstructed_variable: Tensor, original_variable: Tensor, reduction: str) -> Tensor:
        if self.variable_metadata.is_binary():
            return binary_cross_entropy(reconstructed_variable, original_variable, reduction=reduction)
        elif self.variable_metadata.is_numerical():
            return mse_loss(reconstructed_variable, original_variable, reduction=reduction)


class MultiReconstructionLoss(Module):
    metadata: Metadata
    reduction: str

    def __init__(self, metadata: Metadata, reduction: str = "mean"):
        super(MultiReconstructionLoss, self).__init__()
        self.metadata = metadata
        self.reduction = reduction

    @staticmethod
    def _extract_variable(reconstructed: Tensor, original: Tensor, feature_index: int, size: int):
        limit = feature_index + size
        reconstructed_variable = reconstructed[:, feature_index:limit]
        original_variable = original[:, feature_index:limit]
        return reconstructed_variable, original_variable, limit

    def forward(self, reconstructed: Tensor, original: Tensor) -> Tensor:
        loss = 0
        feature_index = 0
        current_block = None

        for variable_metadata in self.metadata.get_by_variable():
            # first check if a block needs to be created
            if current_block is not None and not current_block.matches_type(variable_metadata):
                # extract the original and target variable while moving the feature index
                reconstructed_variable, original_variable, feature_index = self._extract_variable(
                    reconstructed, original, feature_index, current_block.get_size())
                # calculate the block loss
                loss += current_block.build(reconstructed_variable, original_variable, self.reduction)
                # empty the block
                current_block = None

            # if it is a binary or numerical variable
            if variable_metadata.is_binary() or variable_metadata.is_numerical():
                # create a block
                if current_block is None:
                    current_block = BlockLossBuilder(variable_metadata)
                # or add to the existing block
                else:
                    current_block.add(variable_metadata)

            # if it is a categorical variable
            elif variable_metadata.is_categorical():
                # extract the original and target variable while moving the feature index
                reconstructed_variable, original_variable, feature_index = self._extract_variable(
                    reconstructed, original, feature_index, variable_metadata.get_size())
                # calculate the categorical loss
                loss += cross_entropy(reconstructed_variable,
                                      torch.argmax(original_variable, dim=1),  # one-hot to label encoding required
                                      reduction=self.reduction)

            # if it is another type
            else:
                raise Exception("Unexpected variable type '{}' for variable '{}'.".format(
                    variable_metadata.get_type(), variable_metadata.get_name()))

        # if there is still accumulated data for a block
        if current_block is not None:
            # extract the original and target variable while moving the feature index
            reconstructed_variable, original_variable, feature_index = self._extract_variable(
                reconstructed, original, feature_index, current_block.get_size())
            # calculate the block loss
            loss += current_block.build(reconstructed_variable, original_variable, self.reduction)

        return loss


class MultiReconstructionLossFactory(Factory):

    def create(self, architecture: Architecture, metadata: Metadata, global_configuration: Configuration,
               configuration: Configuration) -> Any:
        optional = configuration.get_all_defined(["reduction"])
        return MultiReconstructionLoss(metadata, **optional)

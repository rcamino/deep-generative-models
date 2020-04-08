import torch

from typing import Any, List

from torch import Tensor
from torch.nn import Embedding, Identity, Module

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.layers.embeddings import compute_embedding_size
from deep_generative_models.layers.input_layer import InputLayer
from deep_generative_models.metadata import Metadata
from deep_generative_models.component_factory import ComponentFactory


def concatenate_condition_if_needed(inputs, condition, conditional_layer):
    if conditional_layer is None and condition is not None:
        raise Exception("An unexpected condition was received.")
    elif conditional_layer is not None and condition is None:
        raise Exception("Expected condition not received.")
    elif conditional_layer is not None and condition is not None:
        return torch.cat((inputs, conditional_layer(condition)), dim=1)
    else:
        return inputs


class ConditionalLayer(InputLayer):

    output_size: int
    layer: Module

    def __init__(self, metadata: Metadata, min_embedding_size: int = 2, max_embedding_size: int = 50) -> None:
        super(ConditionalLayer, self).__init__()

        dependent_variable = metadata.get_dependent_variable()

        if dependent_variable.is_binary():
            self.output_size = 1
            self.layer = Identity()
        elif dependent_variable.is_categorical():
            variable_size = dependent_variable.get_size()
            self.output_size = compute_embedding_size(variable_size, min_embedding_size, max_embedding_size)
            self.layer = Embedding(variable_size, self.output_size)
        else:
            raise Exception("Invalid dependent variable type for conditional layer.")

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layer(inputs.view(-1, 1))

    def get_output_size(self) -> int:
        return self.output_size


class ConditionalLayerFactory(ComponentFactory):

    def optional_arguments(self) -> List[str]:
        return ["min_embedding_size", "max_embedding_size"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        return ConditionalLayer(metadata, **arguments.get_all_defined(self.optional_arguments()))

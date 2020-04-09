import torch

from typing import Optional

from torch import Tensor
from torch.nn import Embedding, Identity, Module

from deep_generative_models.layers.embeddings import compute_embedding_size
from deep_generative_models.layers.input_layer import InputLayer
from deep_generative_models.metadata import Metadata


class ConditionalLayer(InputLayer):

    input_layer: InputLayer
    output_size: int
    layer: Module

    def __init__(self, input_layer: InputLayer, metadata: Metadata, min_embedding_size: int = 2,
                 max_embedding_size: int = 50) -> None:
        super(ConditionalLayer, self).__init__()

        self.input_layer = input_layer

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

    def forward(self, inputs: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        if condition is None:
            raise Exception("Expected condition not received.")

        return torch.cat((
            self.input_layer(inputs),
            self.layer(condition.view(-1, 1))
        ), dim=1)

    def get_output_size(self) -> int:
        return self.input_layer.get_output_size() + self.output_size

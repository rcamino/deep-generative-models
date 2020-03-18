import torch

from typing import Dict

from torch import Tensor
from torch.nn import Module, ParameterList, Parameter

from deep_generative_models.metadata import Metadata


class MultiInputLayer(Module):
    metadata: Metadata
    has_categorical: bool
    size: int
    embeddings: ParameterList
    embedding_by_variable: Dict[str, Parameter]

    def __init__(self, metadata: Metadata, min_embedding_size: int = 2, max_embedding_size: int = 50) -> None:
        super(MultiInputLayer, self).__init__()

        self.metadata = metadata

        self.has_categorical = False
        self.size = 0

        # our embeddings need to be referenced like this to be considered in the parameters of this model
        self.embeddings = ParameterList()
        # this reference is for using the embeddings during the forward pass
        self.embedding_by_variable = {}

        for i, variable_metadata in enumerate(self.metadata.get_by_variable()):
            # if it is a numerical variable
            if variable_metadata.is_numerical():
                assert variable_metadata.get_size() == 1
                self.size += 1

            # if it is a categorical variable
            elif variable_metadata.is_categorical():
                variable_size = variable_metadata.get_size()

                # this is an arbitrary rule of thumb taken from several blog posts
                embedding_size = max(min_embedding_size, min(max_embedding_size, int(variable_size / 2)))

                # the embedding is implemented manually to be able to use one hot encoding
                # PyTorch embedding only accepts as input label encoding
                embedding = Parameter(data=torch.Tensor(variable_size, embedding_size).normal_(), requires_grad=True)

                self.embeddings.append(embedding)
                self.embedding_by_variable[variable_metadata.get_name()] = embedding

                self.size += embedding_size
                self.has_categorical = True

            # if it is another type
            else:
                raise Exception("Unexpected variable type '{}' for variable '{}'.".format(
                    variable_metadata.get_type(), variable_metadata.get_name()))

    def forward(self, inputs: Tensor) -> Tensor:
        if self.has_categorical:
            outputs = []
            start = 0
            for variable_metadata in self.metadata.get_by_variable():
                # extract the variable
                end = start + variable_metadata.get_size()
                variable = inputs[:, start:end]

                # if it is a binary or numerical variable leave the input as it is
                if variable_metadata.is_binary() or variable_metadata.is_numerical():
                    outputs.append(variable)
                # if it is a categorical variable use the embedding
                elif variable_metadata.is_categorical():
                    embedding = self.embedding_by_variable[variable_metadata.get_name()]
                    output = torch.matmul(variable, embedding).squeeze(1)
                    outputs.append(output)
                # it should never get to this part
                else:
                    raise Exception("Unexpected variable type '{}' for variable '{}'.".format(
                        variable_metadata.get_type(), variable_metadata.get_name()))

                # move the variable limits
                start = end

            # concatenate all the variable outputs
            return torch.cat(outputs, dim=1)
        else:
            return inputs

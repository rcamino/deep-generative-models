import torch

from typing import Dict, Any, List

from torch import Tensor
from torch.nn import ParameterList, Parameter

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.layers.embeddings import compute_embedding_size
from deep_generative_models.layers.input_layer import InputLayer
from deep_generative_models.metadata import Metadata
from deep_generative_models.component_factory import MultiComponentFactory


class MultiInputLayer(InputLayer):
    metadata: Metadata
    has_categorical: bool
    output_size: int
    embeddings: ParameterList
    embedding_by_variable: Dict[str, Parameter]

    def __init__(self, metadata: Metadata, min_embedding_size: int = 2, max_embedding_size: int = 50) -> None:
        super(MultiInputLayer, self).__init__()

        self.metadata = metadata

        self.has_categorical = False
        self.output_size = 0

        # our embeddings need to be referenced like this to be considered in the parameters of this model
        self.embeddings = ParameterList()
        # this reference is for using the embeddings during the forward pass
        self.embedding_by_variable = {}

        for i, variable_metadata in enumerate(self.metadata.get_by_independent_variable()):
            # if it is a numerical variable
            if variable_metadata.is_binary() or variable_metadata.is_numerical():
                assert variable_metadata.get_size() == 1
                self.output_size += 1

            # if it is a categorical variable
            elif variable_metadata.is_categorical():
                variable_size = variable_metadata.get_size()

                # this is an arbitrary rule of thumb taken from several blog posts
                embedding_size = compute_embedding_size(variable_size, min_embedding_size, max_embedding_size)

                # the embedding is implemented manually to be able to use one hot encoding
                # PyTorch embedding only accepts as input label encoding
                embedding = Parameter(data=torch.Tensor(variable_size, embedding_size).normal_(), requires_grad=True)

                self.embeddings.append(embedding)
                self.embedding_by_variable[variable_metadata.get_name()] = embedding

                self.output_size += embedding_size
                self.has_categorical = True

            # if it is another type
            else:
                raise Exception("Unexpected variable type '{}' for variable '{}'.".format(
                    variable_metadata.get_type(), variable_metadata.get_name()))

    def forward(self, inputs: Tensor, **additional_inputs: Tensor) -> Tensor:
        # this is a "leaf" input layer (no child input layers)
        # so no additional inputs should remain
        for additional_inputs_name, additional_inputs_value in additional_inputs.items():
            if additional_inputs_value is not None:  # sometimes it makes things easier if I pass None
                raise Exception("Unexpected additional inputs received: {}.".format(additional_inputs_name))

        if self.has_categorical:
            outputs = []
            start = 0
            for variable_metadata in self.metadata.get_by_independent_variable():
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

    def get_output_size(self) -> int:
        return self.output_size


class MultiInputLayerFactory(MultiComponentFactory):

    def optional_arguments(self) -> List[str]:
        return ["min_embedding_size", "max_embedding_size"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        return MultiInputLayer(metadata, **arguments.get_all_defined(self.optional_arguments()))

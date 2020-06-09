import torch

import numpy as np

from typing import List, Any

from torch import Tensor

from deep_generative_models.architecture import Architecture
from deep_generative_models.component_factory import ComponentFactory
from deep_generative_models.configuration import Configuration
from deep_generative_models.gpu import to_gpu_if_available
from deep_generative_models.imputation.masks import compose_with_mask
from deep_generative_models.layers.imputation_layer import ImputationLayer
from deep_generative_models.metadata import Metadata


class MeanAndModesImputationLayer(ImputationLayer):
    means_and_modes: Tensor
    differentiable: bool

    def __init__(self, means_and_modes: Tensor, differentiable: bool = True) -> None:
        super(MeanAndModesImputationLayer, self).__init__()
        self.means_and_modes = means_and_modes
        self.differentiable = differentiable

    def forward(self, inputs: Tensor, missing_mask: Tensor) -> Tensor:
        filling_values = self.means_and_modes.repeat(len(inputs), 1)

        return compose_with_mask(missing_mask,
                                 where_one=filling_values,
                                 where_zero=inputs,
                                 differentiable=self.differentiable)


class MeanAndModesImputationLayerFactory(ComponentFactory):

    def mandatory_arguments(self) -> List[str]:
        return ["path"]

    def optional_arguments(self) -> List[str]:
        return ["differentiable"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        return MeanAndModesImputationLayer(to_gpu_if_available(torch.from_numpy(np.load(arguments.path)).float()),
                                           **arguments.get_all_defined(["differentiable"]))

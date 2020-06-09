import argparse
import torch

from typing import List

import numpy as np

from torch import Tensor

from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.imputation.basic_imputation_task import BasicImputation
from deep_generative_models.layers.mean_and_modes_imputation_layer import MeanAndModesImputationLayer
from deep_generative_models.metadata import Metadata


class MeansAndModesImputation(BasicImputation):

    def mandatory_arguments(self) -> List[str]:
        return super(MeansAndModesImputation, self).mandatory_arguments() + ["means_and_modes"]

    def impute(self, configuration: Configuration, metadata: Metadata, scaled_inputs: Tensor, missing_mask: Tensor
               ) -> Tensor:
        # the filling values are expected to be scaled as the inputs
        filling_values = torch.from_numpy(np.load(configuration.means_and_modes))

        # fill where the missing mask is one (this is scaled too)
        return MeanAndModesImputationLayer(filling_values, differentiable=False)(scaled_inputs, missing_mask)


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Impute with the mean or mode.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    MeansAndModesImputation().timed_run(load_configuration(options.configuration))

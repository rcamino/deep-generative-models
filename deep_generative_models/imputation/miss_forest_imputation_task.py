import argparse
import pickle

from typing import List

import numpy as np
import torch

from torch import Tensor

from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.imputation.basic_imputation_task import BasicImputation
from deep_generative_models.imputation.masks import compose_with_mask
from deep_generative_models.metadata import Metadata


class MissForestImputation(BasicImputation):

    def mandatory_arguments(self) -> List[str]:
        return super(MissForestImputation, self).mandatory_arguments() + ["model"]

    def impute(self, configuration: Configuration, metadata: Metadata, scaled_inputs: Tensor, missing_mask: Tensor
               ) -> Tensor:
        with open(configuration.model, "rb") as model_file:
            model = pickle.load(model_file)

        # the model need np.nan in the missing values to work
        scaled_inputs = compose_with_mask(missing_mask,
                                          where_one=torch.empty_like(scaled_inputs).fill_(np.nan),
                                          where_zero=scaled_inputs,
                                          differentiable=False)  # cannot be differentiable with nans!

        # impute with the scikit-learn model
        imputed = model.transform(scaled_inputs)

        # go back to torch (annoying)
        return torch.from_numpy(imputed)


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Impute with Miss Forest.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    MissForestImputation().timed_run(load_configuration(options.configuration))

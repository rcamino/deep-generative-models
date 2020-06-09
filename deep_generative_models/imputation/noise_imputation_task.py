import argparse

from typing import List

from torch import Tensor

from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.imputation.basic_imputation_task import BasicImputation
from deep_generative_models.layers.normal_noise_imputation_layer import NormalNoiseImputationLayer
from deep_generative_models.metadata import Metadata


class NormalNoiseImputation(BasicImputation):

    def optional_arguments(self) -> List[str]:
        return super(NormalNoiseImputation, self).optional_arguments() + ["noise_mean", "noise_std"]

    def impute(self, configuration: Configuration, metadata: Metadata, scaled_inputs: Tensor, missing_mask: Tensor
               ) -> Tensor:
        optional = configuration.get_all_defined(["noise_mean", "noise_std"])
        optional["differentiable"] = False
        return NormalNoiseImputationLayer(**optional)(scaled_inputs, missing_mask)


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Impute with normal noise.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    NormalNoiseImputation().timed_run(load_configuration(options.configuration))

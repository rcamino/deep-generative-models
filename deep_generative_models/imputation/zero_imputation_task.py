import argparse

from torch import Tensor

from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.imputation.basic_imputation_task import BasicImputation
from deep_generative_models.layers.zero_imputation_layer import ZeroImputationLayer
from deep_generative_models.metadata import Metadata


class ZeroImputation(BasicImputation):

    def impute(self, configuration: Configuration, metadata: Metadata, scaled_inputs: Tensor, missing_mask: Tensor
               ) -> Tensor:
        return ZeroImputationLayer(differentiable=False)(scaled_inputs, missing_mask)


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Impute with zeros.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    ZeroImputation().timed_run(load_configuration(options.configuration))

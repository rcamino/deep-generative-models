import argparse
import torch

import numpy as np

from typing import List

from torch.nn.functional import one_hot

from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.imputation.masks import generate_missing_mask_for, compose_with_mask
from deep_generative_models.metadata import load_metadata
from deep_generative_models.tasks.task import Task


class BasicImputation(Task):

    def mandatory_arguments(self) -> List[str]:
        return [
            "metadata",
            "inputs",
            "missing_mask",
            "outputs",
        ]

    def optional_arguments(self) -> List[str]:
        return super(BasicImputation, self).optional_arguments() + ["output_statistics"]

    def run(self, configuration: Configuration) -> None:
        metadata = load_metadata(configuration.metadata)
        inputs = torch.from_numpy(np.load(configuration.inputs))
        missing_mask = torch.from_numpy(np.load(configuration.missing_mask))

        filling_values = torch.zeros(metadata.get_num_features(), dtype=inputs.dtype)
        for variable_metadata in metadata.get_by_independent_variable():
            index = variable_metadata.get_feature_index()
            size = variable_metadata.get_size()
            values = inputs[:, index:index + size]
            # binary
            if variable_metadata.is_binary():
                # how many ones
                one_count = values.sum()
                zero_count = inputs.shape[0] - one_count
                # more ones than zeros
                if one_count >= zero_count:
                    filling_value = 1
                # more zeros than ones
                else:
                    filling_value = 0
            # categorical
            elif variable_metadata.is_categorical():
                # how many ones per column (per categorical variable value)
                column_count = values.sum(dim=0).view(-1)
                # get the most common
                filling_value = one_hot(column_count.argmax(), num_classes=size)
            # numerical
            else:
                filling_value = values.mean()
            # fill the variable
            filling_values[index:index + size] = filling_value

        # only save the filling values
        if configuration.get("output_statistics", False):
            np.save(configuration.outputs, filling_values.numpy())
        # fill where the missing mask is one
        else:
            output = compose_with_mask(missing_mask,
                                       where_one=filling_values.repeat(len(inputs), 1),
                                       where_zero=inputs,
                                       differentiable=False)

            np.save(configuration.outputs, output.numpy())


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Impute with the mean or mode.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    BasicImputation().timed_run(load_configuration(options.configuration))

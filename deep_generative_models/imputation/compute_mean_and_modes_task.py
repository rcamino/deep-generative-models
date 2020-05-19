import argparse
import torch

import numpy as np

from typing import List

from torch.nn.functional import one_hot

from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.imputation.masks import inverse_mask
from deep_generative_models.metadata import load_metadata
from deep_generative_models.tasks.task import Task


class ComputeMeansAndModes(Task):

    def mandatory_arguments(self) -> List[str]:
        return [
            "metadata",
            "inputs",
            "missing_mask",
            "outputs",
        ]

    def run(self, configuration: Configuration) -> None:
        metadata = load_metadata(configuration.metadata)
        inputs = torch.from_numpy(np.load(configuration.inputs))
        missing_mask = torch.from_numpy(np.load(configuration.missing_mask))
        non_missing_mask = inverse_mask(missing_mask)

        assert inputs.shape == missing_mask.shape

        filling_values = torch.zeros(metadata.get_num_features(), dtype=inputs.dtype)
        for variable_metadata in metadata.get_by_independent_variable():
            index = variable_metadata.get_feature_index()
            size = variable_metadata.get_size()
            # binary
            if variable_metadata.is_binary():
                # count how many ones in the variable where the non missing mask is one
                one_count = inputs[non_missing_mask[:, index] == 1, index].sum()
                # count how many ones non missing values the variable has and subtract the ones
                zero_count = non_missing_mask[:, index].sum() - one_count
                # fill with a one if there are more ones than zeros
                # if not fill with a zero
                filling_value = (1 if one_count >= zero_count else 0)
            # categorical
            elif variable_metadata.is_categorical():
                # how many ones per column (per categorical variable value)
                column_count = torch.zeros(size)
                for offset in range(size):
                    column_count[offset] = inputs[non_missing_mask[:, index + offset] == 1, index + offset].sum()
                # get the most common
                filling_value = one_hot(column_count.argmax(), num_classes=size)
            # numerical
            else:
                # take the mean of the values where the non missing mask is one
                filling_value = inputs[non_missing_mask[:, index] == 1, index].mean()
            # fill the variable
            filling_values[index:index + size] = filling_value

        # save the filling values
        np.save(configuration.outputs, filling_values.numpy())


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Calculate the mean or mode per variable.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    ComputeMeansAndModes().timed_run(load_configuration(options.configuration))

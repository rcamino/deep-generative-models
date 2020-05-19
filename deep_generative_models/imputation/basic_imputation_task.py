import argparse
import pickle
import torch

import numpy as np

from csv import DictWriter
from typing import List

from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.imputation.masks import compose_with_mask
from deep_generative_models.losses.rmse import RMSE
from deep_generative_models.tasks.task import Task


class BasicImputation(Task):

    def mandatory_arguments(self) -> List[str]:
        return [
            "inputs",
            "missing_mask",
            "means_and_modes",
        ]

    def optional_arguments(self) -> List[str]:
        return super(BasicImputation, self).optional_arguments() + ["scaler", "outputs", "logs"]

    def run(self, configuration: Configuration) -> None:
        inputs = torch.from_numpy(np.load(configuration.inputs))
        missing_mask = torch.from_numpy(np.load(configuration.missing_mask))
        filling_values = torch.from_numpy(np.load(configuration.means_and_modes))

        # fill where the missing mask is one
        imputed = compose_with_mask(missing_mask,
                                    where_one=filling_values.repeat(len(inputs), 1),
                                    where_zero=inputs,
                                    differentiable=False)

        # scale back if requested
        if "scaler" in configuration:
            with open(configuration.scaler, "rb") as scaler_file:
                scaler = pickle.load(scaler_file)
                inputs = torch.from_numpy(scaler.inverse_transform(inputs.numpy()))
                imputed = torch.from_numpy(scaler.inverse_transform(imputed.numpy()))

        # if imputation should be saved
        if "outputs" in configuration:
            np.save(configuration.outputs, imputed)

        # if reconstruction loss should be logged
        if "logs" in configuration:
            # calculate reconstruction loss
            reconstruction_loss_function = RMSE()
            reconstruction_loss = reconstruction_loss_function(imputed, inputs).item()

            # this uses one row on a csv file
            file_mode = "a" if configuration.logs.get("append", False) else "w"
            with open(configuration.logs.path, file_mode) as reconstruction_loss_file:
                file_writer = DictWriter(reconstruction_loss_file, [
                    "inputs",
                    "missing_mask",
                    "means_and_modes",
                    "reconstruction_loss"
                ])

                # write the csv header if it is the first time
                if not configuration.logs.get("append", False):
                    file_writer.writeheader()

                row = {
                    "inputs": configuration.inputs,
                    "missing_mask": configuration.missing_mask,
                    "means_and_modes": configuration.means_and_modes,
                    "reconstruction_loss": reconstruction_loss,
                }

                self.logger.info(row)
                file_writer.writerow(row)


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Impute with the mean or mode.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    BasicImputation().timed_run(load_configuration(options.configuration))

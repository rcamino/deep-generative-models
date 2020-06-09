import argparse
import os
import torch

import numpy as np

from csv import DictWriter
from typing import List

from torch import Tensor
from torch.nn import MSELoss

from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.imputation.masks import compose_with_mask
from deep_generative_models.losses.multi_reconstruction import MultiReconstructionLoss
from deep_generative_models.losses.rmse import RMSE
from deep_generative_models.metadata import load_metadata
from deep_generative_models.post_processing import load_scale_transform
from deep_generative_models.tasks.task import Task


class BasicImputation(Task):

    def mandatory_arguments(self) -> List[str]:
        return [
            "metadata",
            "inputs",
            "missing_mask",
            "means_and_modes",
        ]

    def optional_arguments(self) -> List[str]:
        return super(BasicImputation, self).optional_arguments() + ["scaler", "outputs", "logs"]

    def run(self, configuration: Configuration) -> None:
        metadata = load_metadata(configuration.metadata)

        # the inputs are expected to be scaled
        scaled_inputs = torch.from_numpy(np.load(configuration.inputs))

        missing_mask = torch.from_numpy(np.load(configuration.missing_mask))
        filling_values = torch.from_numpy(np.load(configuration.means_and_modes))

        # fill where the missing mask is one (this is scaled too)
        scaled_imputed = compose_with_mask(missing_mask,
                                           where_one=filling_values.repeat(len(scaled_inputs), 1),
                                           where_zero=scaled_inputs,
                                           differentiable=False)

        # scale back if requested
        if "scaler" in configuration:
            scale_transform = load_scale_transform(configuration.scaler)
            inputs = torch.from_numpy(scale_transform.inverse_transform(scaled_inputs.numpy()))
            imputed = torch.from_numpy(scale_transform.inverse_transform(scaled_imputed.numpy()))
            outputs = imputed
        # do not scale back
        else:
            inputs = None
            imputed = None
            outputs = scaled_imputed

        # if imputation should be saved
        if "outputs" in configuration:
            np.save(configuration.outputs, outputs)

        # if reconstruction loss should be logged
        if "logs" in configuration:
            # this uses one row on a csv file
            file_mode = "a" if os.path.exists(configuration.logs.path) else "w"
            with open(configuration.logs.path, file_mode) as reconstruction_loss_file:
                file_writer = DictWriter(reconstruction_loss_file, [
                    "inputs",
                    "missing_mask",
                    "means_and_modes",
                    "scaled_mse",
                    "scaled_rmse",
                    "scaled_mr",
                    "mse",
                    "rmse",
                    "mr",
                ])

                # write the csv header if it is the first time
                if file_mode == "w":
                    file_writer.writeheader()

                row = {
                    "inputs": configuration.inputs,
                    "missing_mask": configuration.missing_mask,
                    "means_and_modes": configuration.means_and_modes,
                }

                # loss functions
                mse_loss_function = MSELoss()
                rmse_loss_function = RMSE()
                mr_loss_function = MultiReconstructionLoss(metadata)

                # unscaled metrics
                if imputed is not None and inputs is not None:
                    row["mse"] = mse_loss_function(imputed, inputs).item()
                    row["rmse"] = rmse_loss_function(imputed, inputs).item()
                    row["mr"] = mr_loss_function(imputed, inputs).item()

                # scaled metrics
                row["scaled_mse"] = mse_loss_function(scaled_imputed, scaled_inputs).item()
                row["scaled_rmse"] = rmse_loss_function(scaled_imputed, scaled_inputs).item()
                row["scaled_mr"] = mr_loss_function(scaled_imputed, scaled_inputs).item()

                self.logger.info(row)
                file_writer.writerow(row)


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Impute with the mean or mode.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    BasicImputation().timed_run(load_configuration(options.configuration))

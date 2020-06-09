import os
import torch

import numpy as np

from csv import DictWriter
from typing import List

from torch import Tensor
from torch.nn import MSELoss

from deep_generative_models.configuration import Configuration
from deep_generative_models.losses.multi_reconstruction import MultiReconstructionLoss
from deep_generative_models.losses.rmse import RMSE
from deep_generative_models.metadata import load_metadata, Metadata
from deep_generative_models.post_processing import load_scale_transform, PostProcessing
from deep_generative_models.tasks.task import Task


class BasicImputation(Task):

    def mandatory_arguments(self) -> List[str]:
        return [
            "metadata",
            "inputs",
            "missing_mask",
        ]

    def optional_arguments(self) -> List[str]:
        return super(BasicImputation, self).optional_arguments() + ["scaler", "outputs", "logs"]

    def run(self, configuration: Configuration) -> None:
        metadata = load_metadata(configuration.metadata)

        # the inputs are expected to be scaled
        scaled_inputs = torch.from_numpy(np.load(configuration.inputs))
        missing_mask = torch.from_numpy(np.load(configuration.missing_mask))
        # the imputation will be scaled too
        scaled_imputed = self.impute(configuration, metadata, scaled_inputs, missing_mask)
        # post-process (without scaling back)
        scaled_imputed = PostProcessing(metadata).transform(scaled_imputed)

        # scale back if requested
        if "scaler" in configuration:
            post_processing = PostProcessing(metadata, load_scale_transform(configuration.scaler))
            inputs = post_processing.transform(scaled_inputs)
            imputed = post_processing.transform(scaled_imputed)
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

    def impute(self, configuration: Configuration, metadata: Metadata, scaled_inputs: Tensor, missing_mask: Tensor
               ) -> Tensor:
        raise NotImplementedError

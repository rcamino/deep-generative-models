import os

from typing import List, Any

import numpy as np

import torch

from torch.nn import MSELoss

from deep_generative_models.configuration import load_configuration
from deep_generative_models.losses.multi_reconstruction import MultiReconstructionLoss
from deep_generative_models.losses.rmse import RMSE
from deep_generative_models.metadata import load_metadata
from deep_generative_models.post_processing import load_scale_transform
from deep_generative_models.tasks.multiprocess_runner import MultiProcessTaskWorker


class ImputationWorker(MultiProcessTaskWorker):

    @classmethod
    def output_fields(cls) -> List[str]:
        return [
            "case",
            "missing_type",
            "missing_percentage",
            "model",
            "seed",
            "fold",
            "hyper_parameter_index",
            "scaled_mse",
            "scaled_rmse",
            "scaled_mr",
            "mse",
            "rmse",
            "mr",
        ]

    def process(self, inputs: Any) -> None:
        # prepare the outputs
        outputs = dict(inputs)

        # load the scale transform
        # remove the path from the outputs
        scale_transform = load_scale_transform(outputs.pop("scale_transform"))

        # load the imputation task configuration
        impute_task = load_configuration(outputs.pop("impute_task"))

        # the imputation task output exists
        if os.path.exists(impute_task.arguments.output):
            # losses
            mse_loss_function = MSELoss()
            rmse_loss_function = RMSE()
            mr_loss_function = MultiReconstructionLoss(load_metadata(impute_task.arguments.metadata))

            # load the scaled data
            scaled_inputs = torch.from_numpy(np.load(impute_task.arguments.features))
            scaled_imputed = torch.from_numpy(np.load(impute_task.arguments.output))
            # compute the scaled metrics
            outputs["scaled_mse"] = mse_loss_function(scaled_imputed, scaled_inputs).item()
            outputs["scaled_rmse"] = rmse_loss_function(scaled_imputed, scaled_inputs).item()
            outputs["scaled_mr"] = mr_loss_function(scaled_imputed, scaled_inputs).item()

            # apply the inverse scale transform to recover the original unscaled data
            inputs = torch.from_numpy(scale_transform.inverse_transform(scaled_inputs.numpy()))
            imputed = torch.from_numpy(scale_transform.inverse_transform(scaled_imputed.numpy()))
            # compute the unscaled metrics
            outputs["mse"] = mse_loss_function(imputed, inputs).item()
            outputs["rmse"] = rmse_loss_function(imputed, inputs).item()
            outputs["mr"] = mr_loss_function(imputed, inputs).item()

        # if the task was not run
        else:
            self.logger.info("{} does not exist.".format(impute_task.arguments.output))

        # send the outputs
        self.send_output(outputs)

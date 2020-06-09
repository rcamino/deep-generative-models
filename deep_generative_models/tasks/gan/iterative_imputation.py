import argparse

import torch

from typing import List, Dict

from torch import Tensor, FloatTensor
from torch.optim.adam import Adam

from deep_generative_models.architecture import Architecture
from deep_generative_models.architecture_factory import create_component
from deep_generative_models.commandline import create_parent_directories_if_needed
from deep_generative_models.configuration import load_configuration, Configuration
from deep_generative_models.gpu import to_gpu_if_available, to_cpu_if_was_in_gpu
from deep_generative_models.imputation.masks import inverse_mask
from deep_generative_models.losses.masked_reconstruction_loss import MaskedReconstructionLoss
from deep_generative_models.metadata import Metadata
from deep_generative_models.tasks.impute import Impute
from deep_generative_models.tasks.train_logger import TrainLogger


class GANIterativeImputation(Impute):

    def mandatory_architecture_arguments(self) -> List[str]:
        return ["noise_size"]

    def mandatory_architecture_components(self) -> List[str]:
        return ["generator"]

    def mandatory_arguments(self) -> List[str]:
        return super(GANIterativeImputation, self).mandatory_arguments() + ["noise_learning_rate",
                                                                            "max_iterations",
                                                                            "reconstruction_loss",
                                                                            "logs"]

    def optional_arguments(self) -> List[str]:
        return super(GANIterativeImputation, self).optional_arguments() + ["tolerance", "log_missing_loss"]

    def impute(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
               batch: Dict[str, Tensor]) -> Tensor:
        # loss function
        loss_function = create_component(architecture, metadata, configuration.reconstruction_loss)
        masked_loss_function = MaskedReconstructionLoss(loss_function)
        batch_size = batch["features"].shape[0] * batch["features"].shape[1]
        # we need the non missing mask for the loss
        non_missing_mask = inverse_mask(batch["missing_mask"])

        # initial noise
        noise = to_gpu_if_available(FloatTensor(len(batch["features"]), architecture.arguments.noise_size).normal_())
        noise.requires_grad_()

        # it is not the generator what we are updating
        # it is the noise
        optimizer = Adam([noise], weight_decay=0, lr=configuration.noise_learning_rate)
        architecture.generator.eval()

        # logger
        log_path = create_parent_directories_if_needed(configuration.logs)
        logger = TrainLogger(self.logger, log_path, False)

        # initial generation
        logger.start_timer()
        generated = architecture.generator(noise, condition=batch.get("labels"))

        # iterate until we reach the maximum number of iterations or until the non missing loss is too small
        max_iterations = configuration.max_iterations
        for iteration in range(1, max_iterations + 1):
            # compute the loss on the non-missing values
            non_missing_loss = masked_loss_function(generated, batch["features"], non_missing_mask)
            logger.log(iteration, max_iterations, "non_missing_loss", to_cpu_if_was_in_gpu(non_missing_loss).item())

            # this loss only makes sense if the ground truth is present
            # only used for debugging
            if configuration.get("log_missing_loss", False):
                # this part should not affect the gradient calculation
                with torch.no_grad():
                    missing_loss = masked_loss_function(generated, batch["raw_features"], batch["missing_mask"])
                    logger.log(iteration, max_iterations, "missing_loss", to_cpu_if_was_in_gpu(missing_loss).item())

                    loss = loss_function(generated, batch["raw_features"]) / batch_size
                    logger.log(iteration, max_iterations, "loss", to_cpu_if_was_in_gpu(loss).item())

            # if the generation is good enough we stop
            if to_cpu_if_was_in_gpu(non_missing_loss).item() < configuration.get("tolerance", 1e-5):
                break

            # clear previous gradients
            optimizer.zero_grad()
            # compute the gradients
            non_missing_loss.backward()
            # update the noise
            optimizer.step()

            # generate next
            logger.start_timer()
            generated = architecture.generator(noise, condition=batch.get("labels"))

        return generated


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="GAN iterative imputation.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    GANIterativeImputation().timed_run(load_configuration(options.configuration))

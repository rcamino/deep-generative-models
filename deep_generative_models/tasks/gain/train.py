import argparse
import torch

from torch import Tensor

import numpy as np

from typing import Dict, List, Iterator

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.gpu import to_gpu_if_available, to_cpu_if_was_in_gpu
from deep_generative_models.imputation.masks import compose_with_mask, generate_mask_for, inverse_mask
from deep_generative_models.metadata import Metadata
from deep_generative_models.post_processing import PostProcessing
from deep_generative_models.tasks.train import Train, Datasets, Batch


def generate_hint(missing_mask: Tensor, hint_probability: float, metadata: Metadata) -> Tensor:
    # the GAIN paper goes on and on about using a more complex hint mechanism
    # but then in the online code example they use this technique
    # see: https://github.com/jsyoon0823/GAIN/issues/2

    # create a mask with "hint probability" of having ones
    hint_mask = to_gpu_if_available(generate_mask_for(missing_mask, hint_probability, metadata))
    # leave the mask untouched where there are hints (hint_mask=1)
    # but put zeros where there are no hints (hint_mask=0)
    return missing_mask * hint_mask


class TrainGAIN(Train):

    def mandatory_arguments(self) -> List[str]:
        return super(TrainGAIN, self).mandatory_arguments() + ["discriminator_steps",
                                                               "generator_steps",
                                                               "hint_probability"]

    def mandatory_architecture_components(self) -> List[str]:
        return [
            "generator",
            "discriminator",
            "discriminator_optimizer",
            "discriminator_loss",
            "generator_optimizer",
            "generator_loss",
            "val_loss"
        ]

    def train_epoch(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                    datasets: Datasets, post_processing: PostProcessing) -> Dict[str, float]:
        # train
        architecture.generator.train()
        architecture.discriminator.train()

        # prepare to accumulate losses per batch
        losses_by_batch = {"generator": [], "discriminator": []}

        # prepare datasets
        train_datasets = Datasets({"features": datasets.train_features, "missing_mask": datasets.train_missing_mask})
        val_datasets = Datasets({"features": datasets.val_features, "missing_mask": datasets.val_missing_mask})

        # an epoch will stop at any point if there are no more batches
        # it does not matter if there are models with remaining steps
        data_iterator = self.iterate_datasets(configuration, train_datasets)

        while True:
            try:
                losses_by_batch["discriminator"].extend(
                    self.train_discriminator_steps(configuration, metadata, architecture, data_iterator))

                losses_by_batch["generator"].extend(
                    self.train_generator_steps(configuration, metadata, architecture, data_iterator))
            except StopIteration:
                break

        # loss aggregation
        losses = {}

        if configuration.discriminator_steps > 0:
            losses["discriminator_train_mean_loss"] = np.mean(losses_by_batch["discriminator"]).item()

        if configuration.generator_steps > 0:
            losses["generator_train_mean_loss"] = np.mean(losses_by_batch["generator"]).item()

        # validation
        architecture.generator.eval()

        val_losses_by_batch = []

        for batch in self.iterate_datasets(configuration, val_datasets):
            val_losses_by_batch.append(self.val_batch(architecture, batch, post_processing))

        losses["val_mean_loss"] = np.mean(val_losses_by_batch).item()

        return losses

    def train_discriminator_steps(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                                  batch_iterator: Iterator[Batch]) -> List[float]:
        losses = []
        for _ in range(configuration.discriminator_steps):
            batch = next(batch_iterator)
            loss = self.train_discriminator_step(configuration, metadata, architecture, batch)
            losses.append(loss)
        return losses

    @staticmethod
    def train_discriminator_step(configuration: Configuration, metadata: Metadata, architecture: Architecture,
                                 batch: Batch) -> float:
        # clean previous gradients
        architecture.discriminator_optimizer.zero_grad()

        # generate a batch of fake features with the same size as the real feature batch
        noise = to_gpu_if_available(torch.ones_like(batch["features"]).normal_())
        noisy_features = compose_with_mask(batch["missing_mask"],
                                           differentiable=False,  # maybe there are NaNs in the dataset
                                           where_one=noise,
                                           where_zero=batch["features"])
        generated = architecture.generator(noisy_features, missing_mask=batch["missing_mask"])
        # replace the missing features by the generated
        imputed = compose_with_mask(mask=batch["missing_mask"],
                                    differentiable=True,  # now there are no NaNs and this should be used
                                    where_one=generated,
                                    where_zero=batch["features"])
        imputed = imputed.detach()  # do not propagate to the generator
        # generate hint
        hint = generate_hint(batch["missing_mask"], configuration.hint_probability, metadata)

        # calculate loss
        loss = architecture.discriminator_loss(architecture=architecture,
                                               imputed=imputed,
                                               hint=hint,
                                               missing_mask=batch["missing_mask"])

        # calculate gradients
        loss.backward()

        # update the discriminator weights
        architecture.discriminator_optimizer.step()

        # return the loss
        return to_cpu_if_was_in_gpu(loss).item()

    def train_generator_steps(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                              batch_iterator: Iterator[Batch]) -> List[float]:
        losses = []
        for _ in range(configuration.generator_steps):
            batch = next(batch_iterator)
            loss = self.train_generator_step(configuration, metadata, architecture, batch)
            losses.append(loss)
        return losses

    @staticmethod
    def train_generator_step(configuration: Configuration, metadata: Metadata, architecture: Architecture,
                             batch: Batch) -> float:
        # clean previous gradients
        architecture.generator_optimizer.zero_grad()

        # generate a batch of fake features with the same size as the real feature batch
        noise = to_gpu_if_available(torch.ones_like(batch["features"]).normal_())
        noisy_features = compose_with_mask(batch["missing_mask"],
                                           differentiable=False,  # maybe there are NaNs in the dataset
                                           where_one=noise,
                                           where_zero=batch["features"])
        generated = architecture.generator(noisy_features, missing_mask=batch["missing_mask"])
        # replace the missing features by the generated
        imputed = compose_with_mask(mask=batch["missing_mask"],
                                    differentiable=True,  # now there are no NaNs and this should be used
                                    where_one=generated,
                                    where_zero=batch["features"])
        # generate hint
        hint = generate_hint(batch["missing_mask"], configuration.hint_probability, metadata)

        # calculate loss
        loss = architecture.generator_loss(architecture=architecture,
                                           features=batch["features"],
                                           generated=generated,
                                           imputed=imputed,
                                           hint=hint,
                                           non_missing_mask=inverse_mask(batch["missing_mask"]))

        # calculate gradients
        loss.backward()

        # update the generator weights
        architecture.generator_optimizer.step()

        # return the loss
        return to_cpu_if_was_in_gpu(loss).item()

    @staticmethod
    def val_batch(architecture: Architecture, batch: Batch,  post_processing: PostProcessing) -> float:
        noise = to_gpu_if_available(torch.ones_like(batch["features"]).normal_())
        noisy_features = compose_with_mask(mask=batch["missing_mask"],
                                           differentiable=False,  # maybe there are NaNs in the dataset
                                           where_one=noise,
                                           where_zero=batch["features"])
        generated = architecture.generator(noisy_features, missing_mask=batch["missing_mask"])
        # replace the missing features by the generated
        imputed = compose_with_mask(mask=batch["missing_mask"],
                                    differentiable=False,  # back propagation not needed here
                                    where_one=generated,
                                    where_zero=batch["features"])

        # scale transform might be applied to imputation and ground truth to compute the proper validation loss
        loss = architecture.val_loss(post_processing.transform(imputed),
                                     post_processing.transform(batch["features"]))

        return to_cpu_if_was_in_gpu(loss).item()


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Train GAIN.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TrainGAIN().timed_run(load_configuration(options.configuration))

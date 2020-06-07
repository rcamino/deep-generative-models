import numpy as np

from logging import Logger
from typing import Dict, Iterator, List, Optional

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.metadata import Metadata
from deep_generative_models.post_processing import PostProcessing
from deep_generative_models.pre_processing import PreProcessing
from deep_generative_models.tasks.autoencoder.train import TrainAutoEncoder
from deep_generative_models.tasks.gan.train import TrainGAN
from deep_generative_models.tasks.train import Datasets, Batch


class TrainGANWithAutoencoder(TrainGAN):

    autoencoder_train_task: TrainAutoEncoder

    def __init__(self, logger: Optional[Logger] = None):
        super(TrainGANWithAutoencoder, self).__init__(logger=logger)
        self.autoencoder_train_task = TrainAutoEncoder(logger=logger)

    def mandatory_arguments(self) -> List[str]:
        return super(TrainGANWithAutoencoder, self).mandatory_arguments() + ["autoencoder_steps"]

    def mandatory_architecture_components(self) -> List[str]:
        return super(TrainGANWithAutoencoder, self).mandatory_architecture_components() \
               + self.autoencoder_train_task.mandatory_architecture_components()

    def train_epoch(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                    datasets: Datasets, pre_processing: PreProcessing, post_processing: PostProcessing
                    ) -> Dict[str, float]:
        # train
        architecture.autoencoder.train()
        architecture.generator.train()
        architecture.discriminator.train()

        # prepare to accumulate losses per batch
        losses_by_batch = {"autoencoder": [], "generator": [], "discriminator": []}

        # basic data
        train_datasets = Datasets({"features": datasets.train_features})
        val_datasets = Datasets({"features": datasets.val_features})

        # conditional
        if "conditional" in architecture.arguments:
            train_datasets["labels"] = datasets.train_labels
            val_datasets["labels"] = datasets.val_labels

        # missing mask
        if "train_missing_mask" in datasets:
            train_datasets["missing_mask"] = datasets.train_missing_mask
        if "val_missing_mask" in datasets:
            val_datasets["missing_mask"] = datasets.val_missing_mask

        # an epoch will stop at any point if there are no more batches
        # it does not matter if there are models with remaining steps
        data_iterator = self.iterate_datasets(configuration, train_datasets)

        while True:
            try:
                losses_by_batch["autoencoder"].extend(
                    self.train_autoencoder_steps(configuration, architecture, data_iterator, pre_processing)
                )

                losses_by_batch["discriminator"].extend(
                    self.train_discriminator_steps(configuration, metadata, architecture, data_iterator, pre_processing)
                )

                losses_by_batch["generator"].extend(
                    self.train_generator_steps(configuration, metadata, architecture)
                )
            except StopIteration:
                break

        # loss aggregation
        losses = {}

        if configuration.autoencoder_steps > 0:
            losses["autoencoder_train_mean_loss"] = np.mean(losses_by_batch["autoencoder"]).item()

        if configuration.discriminator_steps > 0:
            losses["discriminator_train_mean_loss"] = np.mean(losses_by_batch["discriminator"]).item()

        if configuration.generator_steps > 0:
            losses["generator_train_mean_loss"] = np.mean(losses_by_batch["generator"]).item()

        # validation
        architecture.autoencoder.eval()

        autoencoder_val_losses_by_batch = []

        for batch in self.iterate_datasets(configuration, val_datasets):
            batch = pre_processing.transform(batch)
            autoencoder_val_losses_by_batch.append(
                self.autoencoder_train_task.val_batch(architecture, batch, post_processing)
            )

        losses["autoencoder_val_mean_loss"] = np.mean(autoencoder_val_losses_by_batch).item()

        return losses

    def train_autoencoder_steps(self, configuration: Configuration, architecture: Architecture,
                                batch_iterator: Iterator[Batch], pre_processing: PreProcessing) -> List[float]:
        losses = []
        for _ in range(configuration.autoencoder_steps):
            batch = pre_processing.transform(next(batch_iterator))
            loss = self.autoencoder_train_task.train_batch(architecture, batch)
            losses.append(loss)
        return losses

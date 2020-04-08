import numpy as np

from torch import Tensor

from torch.utils.data.dataloader import DataLoader

from typing import Dict, Iterator, List

from torch.utils.data.dataset import TensorDataset

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.metadata import Metadata
from deep_generative_models.tasks.autoencoder.train import TrainAutoEncoder
from deep_generative_models.tasks.gan.train import TrainGAN
from deep_generative_models.tasks.train import Datasets


class TrainGANWithAutoencoder(TrainGAN):

    autoencoder_train_task: TrainAutoEncoder

    def __init__(self):
        super(TrainGANWithAutoencoder, self).__init__()
        self.autoencoder_train_task = TrainAutoEncoder()

    def mandatory_arguments(self) -> List[str]:
        return super(TrainGANWithAutoencoder, self).mandatory_arguments() + ["autoencoder_steps"]

    def mandatory_architecture_components(self) -> List[str]:
        return super(TrainGANWithAutoencoder, self).mandatory_architecture_components() \
               + self.autoencoder_train_task.mandatory_architecture_components()

    def train_epoch(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                    datasets: Datasets) -> Dict[str, float]:
        # train
        architecture.autoencoder.train()
        architecture.generator.train()
        architecture.discriminator.train()

        # prepare to accumulate losses per batch
        losses_by_batch = {"autoencoder": [], "generator": [], "discriminator": []}

        # conditional
        if "conditional" in architecture.arguments:
            train_datasets = TensorDataset(datasets.train_features, datasets.train_labels)
        # non-conditional
        else:
            train_datasets = datasets.train_features

        # an epoch will stop at any point if there are no more batches
        # it does not matter if there are models with remaining steps
        data_iterator = iter(DataLoader(train_datasets, batch_size=configuration.batch_size, shuffle=True))

        while True:
            try:
                losses_by_batch["autoencoder"].extend(
                    self.train_autoencoder_steps(configuration, architecture, data_iterator))

                losses_by_batch["discriminator"].extend(
                    self.train_discriminator_steps(configuration, metadata, architecture, data_iterator))

                losses_by_batch["generator"].extend(self.train_generator_steps(configuration, metadata, architecture))
            except StopIteration:
                break

        # loss aggregation
        losses = {"autoencoder_train_mean_loss": np.mean(losses_by_batch["autoencoder"]).item(),
                  "generator_train_mean_loss": np.mean(losses_by_batch["generator"]).item(),
                  "discriminator_train_mean_loss": np.mean(losses_by_batch["discriminator"]).item()}

        # validation
        architecture.autoencoder.eval()

        if "val_features" in datasets:
            autoencoder_val_losses_by_batch = []

            for batch in DataLoader(datasets.val_features, batch_size=configuration.batch_size, shuffle=True):
                autoencoder_val_losses_by_batch.append(self.autoencoder_train_task.val_batch(architecture, batch))

            losses["autoencoder_val_mean_loss"] = np.mean(autoencoder_val_losses_by_batch).item()

        return losses

    def train_autoencoder_steps(self, configuration: Configuration, architecture: Architecture,
                                data_iterator: Iterator[Tensor]) -> List[float]:
        losses = []
        for _ in range(configuration.autoencoder_steps):
            batch = next(data_iterator)
            loss = self.autoencoder_train_task.train_batch(architecture, batch)
            losses.append(loss)
        return losses

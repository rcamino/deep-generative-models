import argparse

import numpy as np

from torch import Tensor
from torch.utils.data.dataloader import DataLoader

from typing import Dict, List

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.gpu import to_cpu_if_was_in_gpu
from deep_generative_models.metadata import Metadata
from deep_generative_models.tasks.train import Train, Datasets


class TrainAutoEncoder(Train):

    def mandatory_architecture_components(self) -> List[str]:
        return [
            "autoencoder",
            "autoencoder_optimizer",
            "reconstruction_loss"
        ]

    def prepare_training(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                         datasets: Datasets) -> None:
        architecture.autoencoder.train()

    def train_epoch(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                    datasets: Datasets) -> Dict[str, float]:
        train_loss_by_batch = []

        for batch in DataLoader(datasets.train_features, batch_size=configuration.batch_size, shuffle=True):
            train_loss_by_batch.append(self.train_batch(architecture, batch))

        losses = {"train_reconstruction_mean_loss": np.mean(train_loss_by_batch).item()}

        if "val_features" in datasets:
            val_loss_by_batch = []

            for batch in DataLoader(datasets.val_features, batch_size=configuration.batch_size, shuffle=True):
                val_loss_by_batch.append(self.val_batch(architecture, batch))

            losses["val_reconstruction_mean_loss"] = np.mean(val_loss_by_batch).item()

        return losses

    @staticmethod
    def train_batch(architecture: Architecture, batch: Tensor) -> float:
        architecture.autoencoder_optimizer.zero_grad()

        outputs = architecture.autoencoder(batch)

        loss = architecture.reconstruction_loss(outputs, batch)
        loss.backward()

        architecture.autoencoder_optimizer.step()

        loss = to_cpu_if_was_in_gpu(loss)
        return loss.item()

    @staticmethod
    def val_batch(architecture: Architecture, batch: Tensor) -> float:
        outputs = architecture.autoencoder(batch)
        loss = architecture.reconstruction_loss(outputs, batch)
        loss = to_cpu_if_was_in_gpu(loss)
        return loss.item()


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Train AutoEncoder.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TrainAutoEncoder().timed_run(load_configuration(options.configuration))

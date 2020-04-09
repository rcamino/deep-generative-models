import argparse

import numpy as np

from typing import Dict, List

from torch.utils.data import TensorDataset, DataLoader

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.gpu import to_cpu_if_was_in_gpu
from deep_generative_models.metadata import Metadata
from deep_generative_models.tasks.train import Train, Datasets, Batch


class TrainAutoEncoder(Train):

    def mandatory_architecture_components(self) -> List[str]:
        return [
            "autoencoder",
            "autoencoder_optimizer",
            "reconstruction_loss"
        ]

    def train_epoch(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                    datasets: Datasets) -> Dict[str, float]:
        architecture.autoencoder.train()

        # conditional
        if "conditional" in architecture.arguments:
            train_datasets = TensorDataset(datasets.train_features, datasets.train_labels)
            val_datasets = TensorDataset(datasets.val_features, datasets.val_labels)
        # non-conditional
        else:
            train_datasets = datasets.train_features
            val_datasets = datasets.val_features

        train_loss_by_batch = []

        for batch in DataLoader(train_datasets, batch_size=configuration.batch_size, shuffle=True):
            train_loss_by_batch.append(self.train_batch(architecture, batch))

        losses = {"train_reconstruction_mean_loss": np.mean(train_loss_by_batch).item()}

        if "val_features" in datasets:
            architecture.autoencoder.eval()

            val_loss_by_batch = []

            for batch in DataLoader(val_datasets, batch_size=configuration.batch_size, shuffle=True):
                val_loss_by_batch.append(self.val_batch(architecture, batch))

            losses["val_reconstruction_mean_loss"] = np.mean(val_loss_by_batch).item()

        return losses

    @staticmethod
    def train_batch(architecture: Architecture, batch: Batch) -> float:
        if "conditional" in architecture.arguments:
            features, condition = batch
        else:
            features = batch
            condition = None

        architecture.autoencoder_optimizer.zero_grad()

        outputs = architecture.autoencoder(features, condition=condition)

        loss = architecture.reconstruction_loss(outputs, features)
        loss.backward()

        architecture.autoencoder_optimizer.step()

        loss = to_cpu_if_was_in_gpu(loss)
        return loss.item()

    @staticmethod
    def val_batch(architecture: Architecture, batch: Batch) -> float:
        if "conditional" in architecture.arguments:
            features, condition = batch
        else:
            features = batch
            condition = None

        outputs = architecture.autoencoder(features, condition=condition)
        loss = architecture.reconstruction_loss(outputs, features)
        loss = to_cpu_if_was_in_gpu(loss)
        return loss.item()


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Train AutoEncoder.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TrainAutoEncoder().timed_run(load_configuration(options.configuration))

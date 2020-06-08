import argparse

import numpy as np

from typing import Dict, List

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.gpu import to_cpu_if_was_in_gpu
from deep_generative_models.metadata import Metadata
from deep_generative_models.post_processing import PostProcessing
from deep_generative_models.pre_processing import PreProcessing
from deep_generative_models.tasks.train import Train, Datasets, Batch


class TrainAutoEncoder(Train):

    def mandatory_architecture_components(self) -> List[str]:
        return [
            "autoencoder",
            "autoencoder_optimizer",
            "reconstruction_loss",
            "val_loss",
        ]

    def train_epoch(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                    datasets: Datasets, pre_processing: PreProcessing, post_processing: PostProcessing
                    ) -> Dict[str, float]:
        architecture.autoencoder.train()

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

        # train by batch
        train_loss_by_batch = []

        for batch in self.iterate_datasets(configuration, train_datasets):
            batch = pre_processing.transform(batch)
            train_loss_by_batch.append(self.train_batch(architecture, batch))

        # loss aggregation
        losses = {"train_reconstruction_mean_loss": np.mean(train_loss_by_batch).item()}

        # validation
        architecture.autoencoder.eval()

        val_loss_by_batch = []

        for batch in self.iterate_datasets(configuration, val_datasets):
            batch = pre_processing.transform(batch)
            val_loss_by_batch.append(self.val_batch(architecture, batch, post_processing))

        losses["val_reconstruction_mean_loss"] = np.mean(val_loss_by_batch).item()

        return losses

    @staticmethod
    def train_batch(architecture: Architecture, batch: Batch) -> float:
        architecture.autoencoder_optimizer.zero_grad()

        outputs = architecture.autoencoder(batch["features"], condition=batch.get("labels"))

        loss = architecture.reconstruction_loss(outputs, batch["features"])
        loss.backward()

        architecture.autoencoder_optimizer.step()

        loss = to_cpu_if_was_in_gpu(loss)
        return loss.item()

    @staticmethod
    def val_batch(architecture: Architecture, batch: Batch, post_processing: PostProcessing) -> float:
        outputs = architecture.autoencoder(batch["features"], condition=batch.get("labels"))
        loss = architecture.val_loss(post_processing, outputs["reconstructed"], batch)
        return to_cpu_if_was_in_gpu(loss).item()


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Train AutoEncoder.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TrainAutoEncoder().timed_run(load_configuration(options.configuration))

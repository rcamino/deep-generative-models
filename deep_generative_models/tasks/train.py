import torch

import numpy as np

from typing import Dict, List

from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from deep_generative_models.architecture import Architecture, ArchitectureConfigurationValidator
from deep_generative_models.checkpoints import Checkpoints
from deep_generative_models.commandline import create_parent_directories_if_needed
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.dictionary import Dictionary
from deep_generative_models.architecture_factory import create_architecture
from deep_generative_models.gpu import to_gpu_if_available
from deep_generative_models.post_processing import load_scale_transform, PostProcessing
from deep_generative_models.rng import seed_all
from deep_generative_models.tasks.train_logger import TrainLogger
from deep_generative_models.metadata import load_metadata, Metadata
from deep_generative_models.tasks.task import Task


# the data loader returns a dictionary even if the datasets iterator returns datasets
# so sadly I need this batch class to be a normal dictionary
Batch = Dict[str, Tensor]


class Datasets(Dictionary[Tensor]):
    pass


class DatasetsIterator(Dataset):
    datasets: Datasets

    def __init__(self, datasets: Datasets) -> None:
        self.datasets = datasets

    def __len__(self) -> int:
        return next(iter(self.datasets.values())).shape[0]

    def __getitem__(self, index) -> Batch:
        indexed = {}
        for key, values in self.datasets.items():
            indexed[key] = values[index]
        return indexed


class Train(Task, ArchitectureConfigurationValidator):

    def mandatory_arguments(self) -> List[str]:
        return [
            "data",
            "metadata",
            "architecture",
            "checkpoints",
            "logs",
            "batch_size",
            "epochs"
        ]

    def optional_arguments(self) -> List[str]:
        return super(Train, self).optional_arguments() + ["seed",
                                                          "scale_transform",
                                                          "keep_checkpoint_by_metric"]

    @staticmethod
    def iterate_datasets(configuration: Configuration, datasets: Datasets):
        return iter(DataLoader(DatasetsIterator(datasets), batch_size=configuration.batch_size, shuffle=True))

    def run(self, configuration: Configuration) -> None:
        seed_all(configuration.get("seed"))

        datasets = Datasets()
        for dataset_name, dataset_path in configuration.data.items():
            datasets[dataset_name] = to_gpu_if_available(torch.from_numpy(np.load(dataset_path)).float())

        metadata = load_metadata(configuration.metadata)

        if "scale_transform" in configuration:
            scale_transform = load_scale_transform(configuration.scale_transform)
        else:
            scale_transform = None

        post_processing = PostProcessing(metadata, scale_transform)

        architecture_configuration = load_configuration(configuration.architecture)
        self.validate_architecture_configuration(architecture_configuration)
        architecture = create_architecture(metadata, architecture_configuration)
        architecture.to_gpu_if_available()

        create_parent_directories_if_needed(configuration.checkpoints.output)
        checkpoints = Checkpoints()

        # no input checkpoint by default
        checkpoint = None

        # continue from an output checkpoint (has priority over input checkpoint)
        if configuration.checkpoints.get("continue_from_output", default=False) \
                and checkpoints.exists(configuration.checkpoints.output):
            checkpoint = checkpoints.load(configuration.checkpoints.output)
        # continue from an input checkpoint
        elif "input" in configuration.checkpoints:
            checkpoint = checkpoints.load(configuration.checkpoints.input)
            if configuration.checkpoints.get("ignore_input_epochs", default=False):
                checkpoint["epoch"] = 0
            if configuration.checkpoints.get("use_best_input", default=False):
                checkpoint["architecture"] = checkpoint.pop("best_architecture")
                checkpoint.pop("best_epoch")
                checkpoint.pop("best_metric")

        # if there is no starting checkpoint then initialize
        if checkpoint is None:
            architecture.initialize()

            checkpoint = {
                "architecture": checkpoints.extract_states(architecture),
                "epoch": 0
            }
        # if there is a starting checkpoint then load it
        else:
            checkpoints.load_states(checkpoint["architecture"], architecture)

        log_path = create_parent_directories_if_needed(configuration.logs)
        logger = TrainLogger(self.logger, log_path, checkpoint["epoch"] > 0)

        for epoch in range(checkpoint["epoch"] + 1, configuration.epochs + 1):
            # train discriminator and generator
            logger.start_timer()

            metrics = self.train_epoch(configuration, metadata, architecture, datasets, post_processing)

            for metric_name, metric_value in metrics.items():
                logger.log(epoch, configuration.epochs, metric_name, metric_value)

            # update the checkpoint
            checkpoint["architecture"] = checkpoints.extract_states(architecture)
            checkpoint["epoch"] = epoch

            # if the best architecture parameters should be kept
            if "keep_checkpoint_by_metric" in configuration:
                # get the metric used to compare checkpoints
                checkpoint_metric = metrics[configuration.keep_checkpoint_by_metric]

                # check if this is the best checkpoint (or the first)
                if "best_metric" not in checkpoint or checkpoint_metric < checkpoint["best_metric"]:
                    checkpoint["best_architecture"] = checkpoint["architecture"]
                    checkpoint["best_epoch"] = epoch
                    checkpoint["best_metric"] = checkpoint_metric

            # save checkpoint
            checkpoints.delayed_save(checkpoint, configuration.checkpoints.output, configuration.checkpoints.max_delay)

        # force save of last checkpoint
        checkpoints.save(checkpoint, configuration.checkpoints.output)

        # finish
        logger.close()

    def train_epoch(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                    datasets: Datasets, post_processing: PostProcessing) -> Dict[str, float]:
        raise NotImplementedError

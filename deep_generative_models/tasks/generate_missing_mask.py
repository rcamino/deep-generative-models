import argparse

import torch

import numpy as np

from typing import List

from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.imputation.masks import generate_missing_mask_for
from deep_generative_models.metadata import load_metadata
from deep_generative_models.rng import seed_all
from deep_generative_models.tasks.task import Task


class GenerateMissingMask(Task):

    def mandatory_arguments(self) -> List[str]:
        return [
            "metadata",
            "missing_probability",
            "inputs",
            "outputs",
        ]

    def optional_arguments(self) -> List[str]:
        return super(GenerateMissingMask, self).optional_arguments() + ["seed"]

    def run(self, configuration: Configuration) -> None:
        seed_all(configuration.get("seed"))
        metadata = load_metadata(configuration.metadata)
        inputs = torch.from_numpy(np.load(configuration.inputs))
        missing_mask = generate_missing_mask_for(inputs, configuration.missing_probability, metadata)
        np.save(configuration.outputs, missing_mask.numpy())


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Generate a mask that indicates missing values.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    GenerateMissingMask().timed_run(load_configuration(options.configuration))

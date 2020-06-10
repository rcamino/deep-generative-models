import argparse
import pickle
import torch

import numpy as np

from typing import List

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import ExtraTreeRegressor

from deep_generative_models.commandline import create_parent_directories_if_needed
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.imputation.masks import compose_with_mask
from deep_generative_models.tasks.task import Task


class TrainMissForest(Task):

    def mandatory_arguments(self) -> List[str]:
        return [
            "inputs",
            "missing_mask",
            "outputs",
        ]

    def optional_arguments(self) -> List[str]:
        return ["seed"]

    def run(self, configuration: Configuration) -> None:
        inputs = torch.from_numpy(np.load(configuration.inputs))
        missing_mask = torch.from_numpy(np.load(configuration.missing_mask))

        assert inputs.shape == missing_mask.shape

        # the model need np.nan in the missing values to work
        inputs = compose_with_mask(missing_mask,
                                   where_one=torch.empty_like(inputs).fill_(np.nan),
                                   where_zero=inputs,
                                   differentiable=False)  # cannot be differentiable with nans!

        # create the model
        model = IterativeImputer(random_state=configuration.get("seed", 0),
                                 estimator=ExtraTreeRegressor(),
                                 missing_values=np.nan)

        # go back to torch (annoying)
        model.fit(inputs.numpy())

        # save the model
        with open(create_parent_directories_if_needed(configuration.outputs), "wb") as model_file:
            pickle.dump(model, model_file)


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Train Miss Forest.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TrainMissForest().timed_run(load_configuration(options.configuration))

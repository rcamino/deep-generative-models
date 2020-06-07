import argparse

from typing import List, Dict

from torch import Tensor

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import load_configuration
from deep_generative_models.tasks.impute import Impute


class ImputeWithGAIN(Impute):

    def mandatory_architecture_components(self) -> List[str]:
        return ["generator"]

    def impute(self, architecture: Architecture, batch: Dict[str, Tensor]) -> Tensor:
        return architecture.generator(batch["features"], missing_mask=batch["missing_mask"])


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Impute with GAIN.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    ImputeWithGAIN().timed_run(load_configuration(options.configuration))

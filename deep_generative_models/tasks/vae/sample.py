import argparse

from typing import List

from torch import Tensor, FloatTensor

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.gpu import to_gpu_if_available
from deep_generative_models.metadata import Metadata
from deep_generative_models.tasks.sample import Sample


class SampleVAE(Sample):

    def mandatory_architecture_arguments(self) -> List[str]:
        return ["code_size"]

    def mandatory_architecture_components(self) -> List[str]:
        return ["autoencoder"]

    def generate_sample(self, configuration: Configuration, metadata: Metadata, architecture: Architecture) -> Tensor:
        code = to_gpu_if_available(FloatTensor(configuration.batch_size, architecture.arguments.code_size).normal_())
        return architecture.autoencoder.decode(code)


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Sample from VAE.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    SampleVAE().timed_run(load_configuration(options.configuration))

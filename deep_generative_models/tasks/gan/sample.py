import argparse

from typing import List, Optional

from torch import Tensor, FloatTensor

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.gpu import to_gpu_if_available
from deep_generative_models.metadata import Metadata
from deep_generative_models.tasks.sample import Sample


class SampleGAN(Sample):

    def mandatory_architecture_components(self) -> List[str]:
        return ["generator"]

    def generate_sample(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                        condition: Optional[Tensor] = None) -> Tensor:
        noise = to_gpu_if_available(FloatTensor(configuration.batch_size, architecture.arguments.noise_size).normal_())
        architecture.generator.eval()
        return architecture.generator(noise, condition=condition)


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Sample from GAN.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    SampleGAN().timed_run(load_configuration(options.configuration))

from typing import List

from torch import Tensor, FloatTensor

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.gpu import to_gpu_if_available
from deep_generative_models.metadata import Metadata
from deep_generative_models.tasks.sample import Sample


class SampleGANWithAutoEncoder(Sample):

    def mandatory_architecture_arguments(self) -> List[str]:
        return ["noise_size"]

    def mandatory_architecture_components(self) -> List[str]:
        return ["generator", "autoencoder"]

    def generate_sample(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                        **additional_inputs: Tensor) -> Tensor:
        noise = to_gpu_if_available(FloatTensor(configuration.batch_size, architecture.arguments.noise_size).normal_())
        architecture.autoencoder.eval()
        architecture.generator.eval()
        code = architecture.generator(noise, **additional_inputs)
        return architecture.autoencoder.decode(code, **additional_inputs)

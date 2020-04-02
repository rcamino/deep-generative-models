from torch import Tensor, FloatTensor

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.gpu import to_gpu_if_available
from deep_generative_models.metadata import Metadata
from deep_generative_models.tasks.sample import Sample


class SampleGANWithAutoEncoder(Sample):

    def generate_sample(self, configuration: Configuration, metadata: Metadata, architecture: Architecture) -> Tensor:
        noise = to_gpu_if_available(FloatTensor(configuration.batch_size, configuration.noise_size).normal_())
        code = architecture.generator(noise)
        return architecture.autoencoder.decode(code)

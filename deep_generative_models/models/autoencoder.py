from typing import Tuple, Any

from torch import Tensor
from torch.nn import Module

from deep_generative_models.configuration import Configuration
from deep_generative_models.metadata import Metadata
from deep_generative_models.factory import MultiFactory
from deep_generative_models.models.decoder import Decoder
from deep_generative_models.models.encoder import Encoder


class AutoEncoder(Module):

    encoder: Encoder
    decoder: Decoder

    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super(AutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        code = self.encode(inputs)
        reconstructed = self.decode(code)
        return code, reconstructed

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def decode(self, code: Tensor) -> Tensor:
        return self.decoder(code)


class SingleVariableAutoEncoderFactory(MultiFactory):

    def create(self, metadata: Metadata, global_configuration: Configuration, configuration: Configuration) -> Any:
        return AutoEncoder(
            self.create_other("SingleInputEncoder", metadata, global_configuration, configuration.encoder),
            self.create_other("SingleOutputDecoder", metadata, global_configuration, configuration.decoder)
        )


class MultiVariableAutoEncoderFactory(MultiFactory):

    def create(self, metadata: Metadata, global_configuration: Configuration, configuration: Configuration) -> Any:
        return AutoEncoder(
            self.create_other("MultiInputEncoder", metadata, global_configuration, configuration.encoder),
            self.create_other("MultiOutputDecoder", metadata, global_configuration, configuration.decoder)
        )

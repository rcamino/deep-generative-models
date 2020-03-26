from typing import Any, Dict

from torch import Tensor
from torch.nn import Module

from deep_generative_models.architecture import Architecture
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

    def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
        outputs = {"code": self.encode(inputs)}
        outputs["reconstructed"] = self.decode(outputs["code"])
        return outputs

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def decode(self, code: Tensor) -> Tensor:
        return self.decoder(code)


class SingleVariableAutoEncoderFactory(MultiFactory):

    def create(self, architecture: Architecture, metadata: Metadata, global_configuration: Configuration,
               configuration: Configuration) -> Any:
        return AutoEncoder(
            self.create_other("SingleInputEncoder", architecture, metadata, global_configuration, configuration.encoder),
            self.create_other("SingleOutputDecoder", architecture, metadata, global_configuration, configuration.decoder)
        )


class MultiVariableAutoEncoderFactory(MultiFactory):

    def create(self, architecture: Architecture, metadata: Metadata, global_configuration: Configuration,
               configuration: Configuration) -> Any:
        return AutoEncoder(
            self.create_other("MultiInputEncoder", architecture, metadata, global_configuration, configuration.encoder),
            self.create_other("MultiOutputDecoder", architecture, metadata, global_configuration, configuration.decoder)
        )

from typing import Any, Dict, List

from torch import Tensor
from torch.nn import Module

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.metadata import Metadata
from deep_generative_models.factory import MultiFactory, Factory
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


class AutoEncoderFactory(MultiFactory):

    encoder_factory_name: str
    decoder_factory_name: str

    def __init__(self, factory_by_name: Dict[str, Factory], encoder_factory_name: str,
                 decoder_factory_name: str) -> None:
        super(AutoEncoderFactory, self).__init__(factory_by_name)
        self.encoder_factory_name = encoder_factory_name
        self.decoder_factory_name = decoder_factory_name

    def mandatory_arguments(self) -> List[str]:
        return ["encoder", "decoder"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        encoder = self.create_other(self.encoder_factory_name, architecture, metadata, arguments.encoder)
        decoder = self.create_other(self.decoder_factory_name, architecture, metadata, arguments.decoder)
        return AutoEncoder(encoder, decoder)

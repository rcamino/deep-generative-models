from typing import Any, Dict, List, Optional

from torch import Tensor
from torch.nn import Module

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.metadata import Metadata
from deep_generative_models.component_factory import MultiComponentFactory, ComponentFactory
from deep_generative_models.models.feed_forward import FeedForward


class AutoEncoder(Module):

    encoder: FeedForward
    decoder: FeedForward

    def __init__(self, encoder: FeedForward, decoder: FeedForward) -> None:
        super(AutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs: Tensor, condition: Optional[Tensor] = None) -> Dict[str, Tensor]:
        outputs = self.encode(inputs)
        outputs["reconstructed"] = self.decode(outputs["code"])
        return outputs

    def encode(self, inputs: Tensor, condition: Optional[Tensor] = None) -> Dict[str, Tensor]:
        return {"code": self.encoder(inputs, condition=condition)}

    def decode(self, code: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        return self.decoder(code, condition=condition)


class AutoEncoderFactory(MultiComponentFactory):

    encoder_factory_name: str
    decoder_factory_name: str

    def __init__(self, factory_by_name: Dict[str, ComponentFactory], encoder_factory_name: str,
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

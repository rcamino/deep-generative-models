from typing import Optional, List, Tuple

from torch import Tensor
from torch.nn import Module

from deep_generative_models.metadata import Metadata
from deep_generative_models.models.decoder import Decoder
from deep_generative_models.models.encoder import Encoder


class AutoEncoder(Module):

    encoder: Encoder
    decoder: Decoder

    def __init__(self, input_size: int, code_size: int, encoder_hidden_sizes: List[int] = (),
                 decoder_hidden_sizes: List[int] = (), metadata: Optional[Metadata] = None,
                 temperature: Optional[float] = None) -> None:

        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(input_size, code_size, hidden_sizes=encoder_hidden_sizes, metadata=metadata)

        self.decoder = Decoder(code_size, input_size, hidden_sizes=decoder_hidden_sizes, metadata=metadata,
                               temperature=temperature)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        code = self.encode(inputs)
        reconstructed = self.decode(code)
        return code, reconstructed

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def decode(self, code: Tensor) -> Tensor:
        return self.decoder(code)

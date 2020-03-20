from typing import Tuple

from torch import Tensor
from torch.nn import Module


class AutoEncoder(Module):

    encoder: Module
    decoder: Module

    def __init__(self, encoder: Module, decoder: Module) -> None:
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

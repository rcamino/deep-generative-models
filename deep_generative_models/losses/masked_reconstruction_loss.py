from typing import List, Any

from torch import Tensor
from torch.nn import Module

from deep_generative_models.architecture import Architecture
from deep_generative_models.component_factory import MultiComponentFactory
from deep_generative_models.configuration import Configuration
from deep_generative_models.metadata import Metadata


class MaskedReconstructionLoss(Module):
    reconstruction_loss: Module

    def __init__(self, reconstruction_loss: Module) -> None:
        super(MaskedReconstructionLoss, self).__init__()
        assert reconstruction_loss.reduction == "sum", "The reconstruction loss should have reduction='sum'."
        self.reconstruction_loss = reconstruction_loss

    def forward(self, inputs: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        return self.reconstruction_loss(mask * inputs, mask * target) / mask.sum()


class MaskedReconstructionLossFactory(MultiComponentFactory):

    def mandatory_arguments(self) -> List[str]:
        return ["reconstruction_loss"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        return MaskedReconstructionLoss(self.create_other(arguments.reconstruction_loss.factory,
                                                          architecture,
                                                          metadata,
                                                          arguments.reconstruction_loss.get("arguments", {})))

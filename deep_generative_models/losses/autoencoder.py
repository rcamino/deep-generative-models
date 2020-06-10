from typing import Dict, Any, List

from torch import Tensor
from torch.nn import Module

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.component_factory import MultiComponentFactory
from deep_generative_models.imputation.masks import inverse_mask
from deep_generative_models.losses.masked_reconstruction_loss import MaskedReconstructionLoss
from deep_generative_models.metadata import Metadata


class AutoEncoderLoss(Module):
    """
    This is a silly wrapper to share the same interface with other forms of auto encoders.
    """

    reconstruction_loss: Module
    masked: bool

    def __init__(self, reconstruction_loss: Module, masked: bool = False):
        super(AutoEncoderLoss, self).__init__()

        if masked:
            self.reconstruction_loss = MaskedReconstructionLoss(reconstruction_loss)
        else:
            self.reconstruction_loss = reconstruction_loss

        self.masked = masked

    def forward(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        if self.masked:
            return self.reconstruction_loss(outputs["reconstructed"], batch["features"],
                                            inverse_mask(batch["missing_mask"]))
        else:
            return self.reconstruction_loss(outputs["reconstructed"], batch["features"])


class AutoEncoderLossFactory(MultiComponentFactory):

    def mandatory_arguments(self) -> List[str]:
        return ["reconstruction_loss"]

    def optional_arguments(self) -> List[str]:
        return ["masked"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        reconstruction_loss = self.create_other(arguments.reconstruction_loss.factory,
                                                architecture,
                                                metadata,
                                                arguments.reconstruction_loss.get("arguments", {}))

        return AutoEncoderLoss(reconstruction_loss, **arguments.get_all_defined(["masked"]))

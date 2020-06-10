import torch

from typing import Dict, Any, List

from torch import Tensor
from torch.nn import Module

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.component_factory import MultiComponentFactory
from deep_generative_models.imputation.masks import inverse_mask
from deep_generative_models.losses.masked_reconstruction_loss import MaskedReconstructionLoss
from deep_generative_models.metadata import Metadata


class VAELoss(Module):
    reconstruction_loss: Module
    masked: bool

    def __init__(self, reconstruction_loss: Module, masked: bool = False):
        super(VAELoss, self).__init__()

        if masked:
            self.reconstruction_loss = MaskedReconstructionLoss(reconstruction_loss)
        else:
            self.reconstruction_loss = reconstruction_loss

        self.masked = masked

    def forward(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        if self.masked:
            reconstruction_loss = self.reconstruction_loss(outputs["reconstructed"], batch["features"],
                                                           inverse_mask(batch["missing_mask"]))
        else:
            reconstruction_loss = self.reconstruction_loss(outputs["reconstructed"], batch["features"])

        kld_loss = - 0.5 * torch.sum(1 + outputs["log_var"] - outputs["mu"].pow(2) - outputs["log_var"].exp())

        return reconstruction_loss + kld_loss


class VAELossFactory(MultiComponentFactory):

    def mandatory_arguments(self) -> List[str]:
        return ["reconstruction_loss"]

    def optional_arguments(self) -> List[str]:
        return ["masked"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        # override the reduction argument
        reconstruction_loss_configuration = arguments.reconstruction_loss.get("arguments", {})
        if "reduction" in reconstruction_loss_configuration:
            assert reconstruction_loss_configuration["reduction"] == "sum"
        else:
            reconstruction_loss_configuration["reduction"] = "sum"

        # create the reconstruction loss
        reconstruction_loss = self.create_other(arguments.reconstruction_loss.factory,
                                                architecture,
                                                metadata,
                                                reconstruction_loss_configuration)

        # create the vae loss
        return VAELoss(reconstruction_loss, **arguments.get_all_defined(["masked"]))

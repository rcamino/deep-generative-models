import torch

from typing import Dict, Any, List

from torch import Tensor
from torch.nn import Module

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.factory import MultiFactory
from deep_generative_models.metadata import Metadata


class VAELoss(Module):

    reconstruction_loss: Module

    def __init__(self, reconstruction_loss: Module):
        super(VAELoss, self).__init__()
        self.reconstruction_loss = reconstruction_loss

    def forward(self, outputs: Dict[str, Tensor], features: Tensor) -> Tensor:
        reconstruction_loss = self.reconstruction_loss(outputs["reconstructed"], features)
        kld_loss = - 0.5 * torch.sum(1 + outputs["log_var"] - outputs["mu"].pow(2) - outputs["log_var"].exp())
        return reconstruction_loss + kld_loss


class VAELossFactory(MultiFactory):

    def mandatory_arguments(self) -> List[str]:
        return ["reconstruction_loss"]

    def create(self, architecture: Architecture, metadata: Metadata, global_configuration: Configuration,
               configuration: Configuration) -> Any:
        # override the reduction argument
        reconstruction_loss_configuration = configuration.reconstruction_loss.get("arguments", {})
        if "reduction" in reconstruction_loss_configuration:
            assert reconstruction_loss_configuration["reduction"] == "sum"
        else:
            reconstruction_loss_configuration["reduction"] = "sum"
        # create the vae loss
        return VAELoss(self.create_other(configuration.reconstruction_loss.factory,
                                         architecture,
                                         metadata,
                                         global_configuration,
                                         reconstruction_loss_configuration))

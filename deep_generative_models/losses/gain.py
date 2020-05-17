from typing import List, Any

from torch import Tensor
from torch.nn import Module, BCELoss

from deep_generative_models.architecture import Architecture
from deep_generative_models.component_factory import MultiComponentFactory
from deep_generative_models.configuration import Configuration
from deep_generative_models.metadata import Metadata


class GAINDiscriminatorLoss(Module):
    bce_loss: BCELoss

    def __init__(self) -> None:
        super(GAINDiscriminatorLoss, self).__init__()
        self.bce_loss = BCELoss()

    def forward(self, architecture: Architecture, imputed: Tensor, hint: Tensor, missing_mask: Tensor) -> Tensor:
        # the discriminator should predict the missing mask
        # which means that it detects which positions where imputed and which ones were real
        predictions = architecture.discriminator(imputed, missing_mask=hint)
        return self.bce_loss(predictions, missing_mask)


class GAINGeneratorLoss(Module):
    reconstruction_loss: Module
    reconstruction_loss_weight: float
    bce_loss: BCELoss

    def __init__(self, reconstruction_loss: Module, reconstruction_loss_weight: float = 1) -> None:
        super(GAINGeneratorLoss, self).__init__()

        assert reconstruction_loss.reduction == "sum", "The reconstruction loss should have reduction='sum'."

        self.reconstruction_loss = reconstruction_loss
        self.reconstruction_loss_weight = reconstruction_loss_weight

        self.bce_loss = BCELoss()

    def forward(self, architecture: Architecture, features: Tensor, generated: Tensor, imputed: Tensor, hint: Tensor,
                non_missing_mask: Tensor) -> Tensor:
        # the discriminator should predict the missing mask
        # which means that it detects which positions where imputed and which ones were real
        predictions = architecture.discriminator(imputed, missing_mask=hint)
        # but the generator wants to fool the discriminator
        # so we optimize for the inverse mask
        adversarial_loss = self.bce_loss(predictions, non_missing_mask)

        # reconstruction of the non-missing values (averaged by the number of non-missing values)
        reconstruction_loss = self.reconstruction_loss(non_missing_mask * generated,
                                                       non_missing_mask * features,
                                                       ) / non_missing_mask.sum()

        # return the complete loss
        return adversarial_loss + self.reconstruction_loss_weight * reconstruction_loss


class GAINGeneratorLossFactory(MultiComponentFactory):

    def mandatory_arguments(self) -> List[str]:
        return ["reconstruction_loss"]

    def optional_arguments(self) -> List[str]:
        return ["reconstruction_loss_weight"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        reconstruction_loss = self.create_other(arguments.reconstruction_loss.factory, architecture, metadata,
                                                arguments.reconstruction_loss.get("arguments", {}))

        optional_arguments = arguments.get_all_defined(["reconstruction_loss_weight"])

        return GAINGeneratorLoss(reconstruction_loss, **optional_arguments)

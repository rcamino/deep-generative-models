from typing import Dict, Any

from torch import Tensor
from torch.nn import Module

from deep_generative_models.configuration import Configuration
from deep_generative_models.factory import MultiFactory
from deep_generative_models.metadata import Metadata


class AutoEncoderLoss(Module):
    """
    This is a silly wrapper to share the same interface with other forms of auto encoders.
    """

    reconstruction_loss: Module

    def __init__(self, reconstruction_loss: Module):
        super(AutoEncoderLoss, self).__init__()
        self.reconstruction_loss = reconstruction_loss

    def forward(self, outputs: Dict[str, Tensor], features: Tensor) -> Tensor:
        return self.reconstruction_loss(outputs["reconstructed"], features)


class AutoEncoderLossFactory(MultiFactory):

    def create(self, metadata: Metadata, global_configuration: Configuration, configuration: Configuration) -> Any:
        return AutoEncoderLoss(self.create_other(configuration.reconstruction_loss.factory,
                                                 metadata,
                                                 global_configuration,
                                                 configuration.reconstruction_loss.get("arguments", {})))

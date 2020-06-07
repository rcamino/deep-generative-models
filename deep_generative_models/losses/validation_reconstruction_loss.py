from typing import List, Any, Dict

from torch import Tensor
from torch.nn import Module

from deep_generative_models.architecture import Architecture
from deep_generative_models.component_factory import MultiComponentFactory
from deep_generative_models.configuration import Configuration
from deep_generative_models.metadata import Metadata
from deep_generative_models.post_processing import PostProcessing


class ValidationReconstructionLoss(Module):
    reconstruction_loss: Module

    def __init__(self, reconstruction_loss: Module) -> None:
        super(ValidationReconstructionLoss, self).__init__()
        self.reconstruction_loss = reconstruction_loss

    def forward(self, post_processing: PostProcessing, prediction: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        return self.reconstruction_loss(post_processing.transform(prediction),
                                        post_processing.transform(batch["features"]))


class ValidationReconstructionLossFactory(MultiComponentFactory):

    def mandatory_arguments(self) -> List[str]:
        return ["reconstruction_loss"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        return ValidationReconstructionLoss(self.create_other(arguments.reconstruction_loss.factory,
                                                              architecture,
                                                              metadata,
                                                              arguments.reconstruction_loss.get("arguments", {})))

from typing import List, Any, Dict

from torch import Tensor
from torch.nn import Module

from deep_generative_models.architecture import Architecture
from deep_generative_models.component_factory import MultiComponentFactory
from deep_generative_models.configuration import Configuration
from deep_generative_models.imputation.masks import compose_with_mask
from deep_generative_models.metadata import Metadata
from deep_generative_models.post_processing import PostProcessing


class ValidationImputationLoss(Module):
    reconstruction_loss: Module

    def __init__(self, reconstruction_loss: Module) -> None:
        super(ValidationImputationLoss, self).__init__()
        self.reconstruction_loss = reconstruction_loss

    def forward(self, post_processing: PostProcessing, prediction: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        imputed = compose_with_mask(mask=batch["missing_mask"],
                                    differentiable=False,  # back propagation not needed here
                                    where_one=prediction,
                                    where_zero=batch["raw_features"])

        return self.reconstruction_loss(post_processing.transform(imputed),
                                        post_processing.transform(batch["raw_features"]))


class ValidationImputationLossFactory(MultiComponentFactory):

    def mandatory_arguments(self) -> List[str]:
        return ["reconstruction_loss"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        return ValidationImputationLoss(self.create_other(arguments.reconstruction_loss.factory,
                                                          architecture,
                                                          metadata,
                                                          arguments.reconstruction_loss.get("arguments", {})))

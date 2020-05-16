from torch import Tensor
from torch.nn import Module

from deep_generative_models.architecture import Architecture


def critic_loss_function(predictions: Tensor) -> Tensor:
    return predictions.mean(0).view(1)


class WGANCriticLoss(Module):

    def forward(self, architecture: Architecture, real_features: Tensor, fake_features: Tensor,
                **additional_inputs: Tensor) -> Tensor:
        # real loss
        real_predictions = architecture.discriminator(real_features, **additional_inputs)
        real_loss = - critic_loss_function(real_predictions)

        # fake loss
        fake_predictions = architecture.discriminator(fake_features, **additional_inputs)
        fake_loss = critic_loss_function(fake_predictions)

        # total loss
        return real_loss + fake_loss


class WGANGeneratorLoss(Module):

    def forward(self, architecture: Architecture, fake_features: Tensor, **additional_inputs: Tensor) -> Tensor:
        fake_predictions = architecture.discriminator(fake_features, **additional_inputs)
        return - critic_loss_function(fake_predictions)

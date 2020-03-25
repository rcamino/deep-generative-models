from torch import Tensor
from torch.nn import Module


def critic_loss_function(predictions: Tensor) -> Tensor:
    return predictions.mean(0).view(1)


class WGANCriticLoss(Module):

    def forward(self, discriminator: Module, real_features: Tensor, fake_features: Tensor) -> Tensor:
        # real loss
        real_predictions = discriminator(real_features)
        real_loss = - critic_loss_function(real_predictions)

        # fake loss
        fake_predictions = discriminator(fake_features)
        fake_loss = critic_loss_function(fake_predictions)

        # total loss
        return real_loss + fake_loss


class WGANGeneratorLoss(Module):

    def forward(self, discriminator: Module, fake_features: Tensor) -> Tensor:
        fake_predictions = discriminator(fake_features)
        return - critic_loss_function(fake_predictions)

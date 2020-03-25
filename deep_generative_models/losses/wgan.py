from typing import Optional, Any

from torch import Tensor, rand, ones_like
from torch.autograd import grad
from torch.nn import Module

from deep_generative_models.configuration import Configuration
from deep_generative_models.factory import Factory
from deep_generative_models.gpu import to_cpu_if_was_in_gpu
from deep_generative_models.metadata import Metadata


def critic_loss_function(predictions: Tensor) -> Tensor:
    return predictions.mean(0).view(1)


class WGANGradientPenalty(Module):
    penalty: float

    def __init__(self, penalty: float) -> None:
        super(WGANGradientPenalty, self).__init__()
        self.penalty = penalty

    def forward(self, discriminator: Module, real_features: Tensor, fake_features: Tensor) -> Tensor:
        alpha = rand(len(real_features), 1)
        alpha = alpha.expand(real_features.size())
        alpha = to_cpu_if_was_in_gpu(alpha)

        interpolates = alpha * real_features + ((1 - alpha) * fake_features)
        interpolates.requires_grad_()
        discriminator_interpolates = discriminator(interpolates)

        gradients = grad(outputs=discriminator_interpolates,
                         inputs=interpolates,
                         grad_outputs=to_cpu_if_was_in_gpu(ones_like(discriminator_interpolates)),
                         create_graph=True,
                         retain_graph=True,
                         only_inputs=True)[0]

        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.penalty


class WGANCriticLoss(Module):
    gradient_penalty_loss: Optional[WGANGradientPenalty]

    def __init__(self, gradient_penalty: Optional[float] = None) -> None:
        super(WGANCriticLoss, self).__init__()
        if gradient_penalty is None:
            self.gradient_penalty_loss = None
        else:
            self.gradient_penalty_loss = WGANGradientPenalty(gradient_penalty)

    def forward(self, discriminator: Module, real_features: Tensor, fake_features: Tensor) -> Tensor:
        # real loss
        real_predictions = discriminator(real_features)
        real_loss = - critic_loss_function(real_predictions)

        # fake loss
        fake_predictions = discriminator(fake_features)
        fake_loss = critic_loss_function(fake_predictions)

        # total loss without gradient penalty
        if self.gradient_penalty_loss is None:
            return real_loss + fake_loss
        # total loss with gradient penalty
        else:
            return real_loss + fake_loss + self.gradient_penalty_loss(discriminator, real_features, fake_features)


class WGANCriticLossFactory(Factory):

    def create(self, metadata: Metadata, global_configuration: Configuration, configuration: Configuration) -> Any:
        return WGANCriticLoss(**global_configuration.get_all_defined(["gradient_penalty"]))


class WGANGeneratorLoss(Module):

    def forward(self, discriminator: Module, fake_features: Tensor) -> Tensor:
        fake_predictions = discriminator(fake_features)
        return - critic_loss_function(fake_predictions)


class WGANGeneratorLossFactory(Factory):

    def create(self, metadata: Metadata, global_configuration: Configuration, configuration: Configuration) -> Any:
        return WGANGeneratorLoss()

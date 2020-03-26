from typing import Any

from torch import Tensor, rand, ones_like
from torch.autograd import grad
from torch.nn import Module

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.factory import Factory
from deep_generative_models.gpu import to_cpu_if_was_in_gpu
from deep_generative_models.losses.wgan import WGANCriticLoss
from deep_generative_models.metadata import Metadata


class WGANCriticLossWithGradientPenalty(WGANCriticLoss):
    weight: float

    def __init__(self, weight: float) -> None:
        super(WGANCriticLossWithGradientPenalty, self).__init__()
        self.weight = weight

    def forward(self, discriminator: Module, real_features: Tensor, fake_features: Tensor) -> Tensor:
        loss = super(WGANCriticLossWithGradientPenalty, self).forward(discriminator, real_features, fake_features)

        # calculate gradient penalty
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

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.weight

        # return total loss
        return loss + gradient_penalty


class WGANCriticLossWithGradientPenaltyFactory(Factory):

    def create(self, architecture: Architecture, metadata: Metadata, global_configuration: Configuration,
               configuration: Configuration) -> Any:
        return WGANCriticLossWithGradientPenalty(configuration.weight)

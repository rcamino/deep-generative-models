from typing import Optional

from torch import Tensor, rand, ones_like
from torch.autograd import grad

from deep_generative_models.architecture import Architecture
from deep_generative_models.gpu import to_cpu_if_was_in_gpu, to_gpu_if_available
from deep_generative_models.losses.wgan import WGANCriticLoss


class WGANCriticLossWithGradientPenalty(WGANCriticLoss):
    weight: float

    def __init__(self, weight: float = 1.0) -> None:
        super(WGANCriticLossWithGradientPenalty, self).__init__()
        self.weight = weight

    def forward(self, architecture: Architecture, real_features: Tensor, fake_features: Tensor,
                condition: Optional[Tensor] = None) -> Tensor:
        loss = super(WGANCriticLossWithGradientPenalty, self).forward(
            architecture, real_features, fake_features, condition=condition)

        # calculate gradient penalty
        alpha = rand(len(real_features), 1)
        alpha = alpha.expand(real_features.size())
        alpha = to_gpu_if_available(alpha)

        interpolates = alpha * real_features + ((1 - alpha) * fake_features)
        interpolates.requires_grad_()

        # we do not interpolate the conditions because they are the same for fake and real features
        discriminator_interpolates = architecture.discriminator(interpolates, condition=condition)

        gradients = grad(outputs=discriminator_interpolates,
                         inputs=interpolates,
                         grad_outputs=to_gpu_if_available(ones_like(discriminator_interpolates)),
                         create_graph=True,
                         retain_graph=True,
                         only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.weight

        # return total loss
        return loss + gradient_penalty

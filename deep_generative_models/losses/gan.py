from torch import Tensor, FloatTensor, zeros, ones
from torch.nn import Module, BCELoss

from deep_generative_models.architecture import Architecture
from deep_generative_models.gpu import to_gpu_if_available


def generate_positive_labels(size: int, smooth: bool):
    if smooth:
        return to_gpu_if_available(FloatTensor(size).uniform_(0.9, 1))
    else:
        return to_gpu_if_available(ones(size))


class GANDiscriminatorLoss(Module):
    smooth_positive_labels: bool
    bce_loss: BCELoss

    def __init__(self, smooth_positive_labels: bool = False) -> None:
        super(GANDiscriminatorLoss, self).__init__()
        self.smooth_positive_labels = smooth_positive_labels

        self.bce_loss = BCELoss()

    def forward(self, architecture: Architecture, real_features: Tensor, fake_features: Tensor,
                **additional_inputs: Tensor) -> Tensor:
        # real loss
        real_predictions = architecture.discriminator(real_features, **additional_inputs)
        positive_labels = generate_positive_labels(len(real_predictions), self.smooth_positive_labels)
        real_loss = self.bce_loss(real_predictions, positive_labels)

        # fake loss
        fake_predictions = architecture.discriminator(fake_features, **additional_inputs)
        negative_labels = to_gpu_if_available(zeros(len(fake_predictions)))
        fake_loss = self.bce_loss(fake_predictions, negative_labels)

        # total loss
        return real_loss + fake_loss


class GANGeneratorLoss(Module):
    smooth_positive_labels: bool
    bce_loss: BCELoss

    def __init__(self, smooth_positive_labels: bool = False) -> None:
        super(GANGeneratorLoss, self).__init__()
        self.smooth_positive_labels = smooth_positive_labels

        self.bce_loss = BCELoss()

    def forward(self, architecture: Architecture, fake_features: Tensor, **additional_inputs: Tensor) -> Tensor:
        fake_predictions = architecture.discriminator(fake_features, **additional_inputs)
        positive_labels = generate_positive_labels(len(fake_predictions), self.smooth_positive_labels)
        return self.bce_loss(fake_predictions, positive_labels)

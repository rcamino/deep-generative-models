import torch

from torch import Tensor

from deep_generative_models.layers.input_layer import InputLayer


class MiniBatchAveraging(InputLayer):
    input_layer: InputLayer

    def __init__(self, input_layer: InputLayer) -> None:
        super(MiniBatchAveraging, self).__init__()
        self.input_layer = input_layer

    def forward(self, inputs: Tensor, **additional_inputs: Tensor) -> Tensor:
        inputs = self.input_layer(inputs, **additional_inputs)
        mean_per_feature = torch.mean(inputs, 0)
        mean_per_feature_repeated = mean_per_feature.repeat(len(inputs), 1)
        return torch.cat((inputs, mean_per_feature_repeated), 1)

    def get_output_size(self) -> int:
        return self.input_layer.get_output_size() * 2

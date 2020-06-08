import torch

from torch import Tensor

from deep_generative_models.layers.input_layer import InputLayer


class GAINInputLayer(InputLayer):
    input_layer: InputLayer
    mask_size: int

    def __init__(self, input_layer: InputLayer, mask_size: int) -> None:
        super(GAINInputLayer, self).__init__()
        self.input_layer = input_layer
        self.mask_size = mask_size

    def forward(self, inputs: Tensor, **additional_inputs: Tensor) -> Tensor:
        if "missing_mask" not in additional_inputs:
            raise Exception("Expected mask not received.")

        # consume the condition additional inputs
        missing_mask = additional_inputs.pop("missing_mask")

        # validate mask size
        assert missing_mask.shape == inputs.shape

        return torch.cat((
            self.input_layer(inputs, **additional_inputs),
            missing_mask
        ), dim=1)

    def get_output_size(self) -> int:
        return self.input_layer.get_output_size() + self.mask_size

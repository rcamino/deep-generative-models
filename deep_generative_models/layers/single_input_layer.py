from torch import Tensor

from deep_generative_models.layers.input_layer import InputLayer


class SingleInputLayer(InputLayer):
    output_size: int

    def __init__(self, input_size) -> None:
        super(SingleInputLayer, self).__init__()
        self.output_size = input_size

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs

    def get_output_size(self) -> int:
        return self.output_size

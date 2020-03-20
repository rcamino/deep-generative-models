from torch.nn import Module


class InputLayer(Module):

    def get_output_size(self) -> int:
        raise NotImplementedError

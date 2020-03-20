from torch.nn import Module


class OutputLayerFactory:

    def create(self, input_size: int) -> Module:
        raise NotImplementedError

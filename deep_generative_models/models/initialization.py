from torch.nn import Module, Linear, BatchNorm1d
from torch.nn.init import xavier_normal_, constant_, normal_


def initialize_module(module: Module):
    if type(module) == Linear:
        xavier_normal_(module.weight)
        if module.bias is not None:
            constant_(module.bias, 0.0)
    elif type(module) == BatchNorm1d:
        normal_(module.weight, 1.0, 0.02)
        constant_(module.bias, 0.0)

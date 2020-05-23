from typing import Optional

import torch

from torch import Tensor

from deep_generative_models.metadata import Metadata


def inverse_mask(mask: Tensor) -> Tensor:
    return torch.ones_like(mask) - mask


def fill_where_mask_is_one(inputs: Tensor, filling_mask: Tensor, value: float) -> Tensor:
    # using multiplication with the mask does not work if the inputs contain NaNs
    outputs = inputs.clone().detach()
    outputs[filling_mask == 1] = value
    return outputs


def compose_with_mask(mask: Tensor, where_one: Optional[Tensor] = None, where_zero: Optional[Tensor] = None,
                      differentiable: bool = False) -> Tensor:
    # use differentiable when NaNs are NOT present in the sources (also required for training)
    if differentiable:
        # in this case both sources need to be present
        assert where_one is not None
        assert where_zero is not None

        return mask * where_one + inverse_mask(mask) * where_zero

    # use non-differentiable when NaNs are present in the sources
    else:
        # the results starts with zeros
        composed = torch.zeros_like(mask)

        # to fill the result where the mask is one
        if where_one is not None:
            # check shape
            assert where_one.shape == mask.shape
            # put zeros in the source where the mask is zero (because we want the values where the mask is one)
            # and add the to the result
            composed += fill_where_mask_is_one(where_one, inverse_mask(mask), 0.0)

        # to fill the result where the mask is zero
        if where_zero is not None:
            # check shape
            assert where_zero.shape == mask.shape
            # put zeros in the source where the mask is one (because we want the values where the mask is zero)
            # and add the to the result
            composed += fill_where_mask_is_one(where_zero, mask, 0.0)

        return composed


def generate_mask_for(source: Tensor, probability: float, metadata: Metadata) -> Tensor:
    variable_masks = []

    # for each variable
    for variable_metadata in metadata.get_by_independent_variable():
        # ones are generated with the indicated probability
        # zeros are generated with the complement of the indicated probability
        variable_mask = (torch.zeros(len(source), 1).uniform_(0.0, 1.0) < probability).float()

        # repeat across all the features if the variable has more than one feature (e.g. one-hot-encoded)
        if variable_metadata.get_size() > 1:
            variable_mask = variable_mask.repeat(1, variable_metadata.get_size())

        # add the variable mask
        variable_masks.append(variable_mask)

    # return the concatenation of each variable mask
    return torch.cat(variable_masks, dim=1)

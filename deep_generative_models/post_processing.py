from torch import Tensor

from torch.nn.functional import one_hot

from deep_generative_models.metadata import Metadata


def post_process_discrete(sample: Tensor, metadata: Metadata) -> Tensor:
    for variable_metadata in metadata.get_by_independent_variable():
        index = variable_metadata.get_feature_index()
        size = variable_metadata.get_size()
        old_value = sample[:, index:index + size]
        # discrete binary
        if variable_metadata.is_binary():
            new_value = (old_value.view(-1) > .5).view(-1, 1)
        # discrete categorical
        elif variable_metadata.is_categorical():
            new_value = one_hot(old_value.argmax(dim=1), num_classes=size)
        # leave numerical variables the untouched
        else:
            new_value = old_value
        sample[:, index:index + size] = new_value
    return sample

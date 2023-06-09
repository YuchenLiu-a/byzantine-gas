from __future__ import annotations

import torch

class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}

    def register(self, target):
        def add_item(key, value):
            if not callable(value):
                raise Exception(f"Error:{value} must be callable!")
            if key in self._dict:
                print(f"\033[31mWarning:\033[0m {value.__name__} already exists and will be overwritten!")
            self[key] = value
            return value

        if callable(target):
            return add_item(target.__name__, target)
        else:
            return lambda x : add_item(target, x)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

def flatten_models(models: list[torch.nn.Module]):
    flat_model_lt = []
    name_shape_tuples = None

    for model in models:
        flat_state_dict, name_shape_tuples = flatten_model(model)
        flat_model_lt.append(flat_state_dict)
    
    flat_models = torch.stack(flat_model_lt)
    struct = {
        'name_shape_tuples': name_shape_tuples
    }

    return flat_models, struct

def flatten_model(model: torch.nn.Module):
    flat_params = []
    name_shape_tuples = []
    for name, param in model.state_dict().items():
        flat_param = param.view(-1)
        flat_params.append(flat_param)
        name_shape_tuples.append((name, param.shape))
    flat_model = torch.cat(flat_params)
    return flat_model, name_shape_tuples

def unflatten_tensor(flat_tensor, struct):
    name_shape_tuples = struct['name_shape_tuples']
    split_size = [t[1].numel() for t in name_shape_tuples]
    split_tensors = torch.split(flat_tensor, split_size)
    state_dict = {name: split_tensor.view(shape) for (name, shape), split_tensor in zip(name_shape_tuples, split_tensors)}
    return state_dict
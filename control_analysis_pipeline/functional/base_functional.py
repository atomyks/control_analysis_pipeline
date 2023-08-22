import torch.nn as nn
import torch

# from nongradient_parameter import NongradParameter


class BaseNongradModel(nn.Module):
    def __init__(self):
        super(BaseNongradModel, self).__init__()
        self._nongrad_params = {}
        self._nongrad_params_flat = {}
        self._models = {}

    def get_nongrad_param_names(self):
        for name in list(self._nongrad_params.keys()):
            yield name

    def register_nongrad_parameter(self, name, value):
        self._nongrad_params[name] = value

    def register_model(self, name, value):
        self._models[name] = value

    def gen_nongrad_params_flat(self):
        print(f"Module params {self._nongrad_params}")
        self._nongrad_params_flat = self._nongrad_params
        for model_key in list(self._models.keys()):
            print(f"Model key: {model_key}")
            submodel_nongrad_params = self._models[model_key].gen_nongrad_params_flat()
            print(submodel_nongrad_params)
            for submodel_key in list(submodel_nongrad_params.keys()):
                flat_key = model_key + "." + submodel_key
                self._nongrad_params_flat[flat_key] = submodel_nongrad_params[submodel_key]
        return self._nongrad_params_flat

    def get_search_space(self):
        search_space = {}
        optim_params = self.gen_nongrad_params_flat()
        for name in list(optim_params.keys()):
            search_space[name] = torch.arange(optim_params[name].lb,
                                              optim_params[name].ub,
                                              optim_params[name].precision)
        return search_space, optim_params

    def __repr__(self):
        report = "\n----------------------------------\n"
        report += f"Nongrad params: {self._nongrad_params.keys()}\n"
        report += f"Registered models: {self._models.keys()}\n"
        return report

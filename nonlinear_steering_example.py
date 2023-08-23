import torch
import torch.nn as nn
import torch.nn.functional as F
import queue
from control_analysis_pipeline.model.model import Model
from control_analysis_pipeline.functional.deadzone import Deadzone


class NonlinearSteering(Model):
    def __init__(self):
        super(NonlinearSteering, self).__init__()

        self.layer1 = Deadzone()
        self.register_model("layer1", self.layer1)

    def forward(self, u_input: torch.tensor):
        y_output = self.layer1(u_input)
        return y_output


if __name__ == "__main__":
    model = NonlinearSteering()
    print("--------------------------TEST--------------------------")
    search_space, nongrad_params_flat = model.get_search_space()

    dataset = [(torch.tensor(-10.), torch.tensor(-10.)),
               (torch.tensor(-55.), torch.tensor(-55.)),
               (torch.tensor(-10.), torch.tensor(-10.)),
               (torch.tensor(-8.), torch.tensor(-8.)),
               (torch.tensor(-5.), torch.tensor(-5.)),
               (torch.tensor(-2.2), torch.tensor(-2.2)),
               (torch.tensor(-2.), torch.tensor(0.)),
               (torch.tensor(-1.), torch.tensor(0.)),
               (torch.tensor(0.), torch.tensor(0.)),
               (torch.tensor(2.), torch.tensor(0.)),
               (torch.tensor(5.), torch.tensor(0.)),
               (torch.tensor(8.), torch.tensor(0.)),
               (torch.tensor(8.2), torch.tensor(8.5)),
               (torch.tensor(10.), torch.tensor(10.)),
               (torch.tensor(11.), torch.tensor(11.)),
               (torch.tensor(12.), torch.tensor(11.)),
               (torch.tensor(100.), torch.tensor(100.)),
               ]

    loss_fn = nn.L1Loss()


    def objective_function(para):
        loss = 0.0
        for name in list(nongrad_params_flat.keys()):
            nongrad_params_flat[name].set(para[name])
        for input, target in dataset:
            y = model(input)
            loss += loss_fn(y, target)
        score = -loss
        return score


    from gradient_free_optimizers import EvolutionStrategyOptimizer

    opt = EvolutionStrategyOptimizer(search_space)
    opt.search(objective_function, n_iter=1000)

    for name in list(nongrad_params_flat.keys()):
        nongrad_params_flat[name].set(opt.best_para[name])

    # Visualize model as json
    import json
    json_dict = model.get_json_repr()
    print(json.dumps(json_dict, sort_keys=False, indent=4))
    
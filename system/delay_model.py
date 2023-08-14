import torch
import torch.nn as nn
import torch.nn.functional as F
import queue


class DelayModel(nn.Module):
    """
    System delay module. This cannot use backpropagation. This can be trained only using non-gradient optimization
    """
    def __init__(self, batch_size=1, num_inputs=1):
        super(DelayModel, self).__init__()

        self.back_prop = False
        self.nongrad_params = {
            "delay": {
                "lb": 0,
                "ub": 10,
                "val": 10,
            }
        }
        self.input_queue = None

        self.batch_size = batch_size
        self.num_inputs = num_inputs
        self.reset()

    def forward(self, u_input: torch.tensor):
        if self.nongrad_params["delay"]["val"] > 0:
            y_output = self.input_queue.get()
            self.input_queue.put(u_input.reshape((self.batch_size, self.num_inputs)))
        else:
            y_output = u_input.reshape((self.batch_size, self.num_inputs))
        return y_output

    def reset(self):
        self.input_queue = queue.Queue(maxsize=self.nongrad_params["delay"]["val"])
        for i in range(self.nongrad_params["delay"]["val"]):
            # u = torch.zeros((self.batch_size, self.num_inputs))
            self.input_queue.put(torch.tensor([torch.nan], dtype=torch.double))

import torch
import torch.nn as nn
import torch.nn.functional as F


class ErrorModelDemo(nn.Module):
    """
    Similar to https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html
    """
    def __init__(self, num_inputs=1, num_outputs=1):
        super(ErrorModelDemo, self).__init__()

        self.back_prop = False
        self.nongrad_params = None

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.tanh = nn.Tanh()
        self.lin1 = nn.Linear(num_inputs + num_outputs, 10, dtype=torch.double)
        self.lin2 = nn.Linear(10, num_outputs, dtype=torch.double)

    def forward(self, u_input: torch.tensor, y_last: torch.tensor):
        """
        :param u_input: torch.tensor, BATCH x INPUTS, system input
        :param y_last: torch.tensor, BATCH x STATES, system input
        :return:
        """
        x1 = self.tanh(self.lin1(torch.cat((u_input, y_last), dim=-1)))
        y_output = self.tanh(self.lin2(x1))
        return y_output

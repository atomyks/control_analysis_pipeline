import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLinearModel(nn.Module):
    """
    Similar to https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html
    """

    def __init__(self, num_inputs=1, num_outputs=1):
        super(BaseLinearModel, self).__init__()

        self.back_prop = True
        self.nongrad_params = None

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.A = nn.Linear(num_outputs, num_outputs, bias=False, dtype=torch.double)
        self.B = nn.Linear(num_inputs, num_outputs, bias=False, dtype=torch.double)

    def forward(self, u_now: torch.tensor, y_prev: torch.tensor):
        # x[k+1] = Ax[k] + Bu[k]
        y_now = self.A(y_prev) + self.B(u_now)
        return y_now

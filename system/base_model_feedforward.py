import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseFeedforwardModel(nn.Module):
    """
    Similar to https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html
    """

    def __init__(self, num_inputs=1, num_outputs=1):
        super(BaseFeedforwardModel, self).__init__()

        self.back_prop = True
        self.nongrad_params = None

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.B = torch.eye(num_outputs, num_inputs, dtype=torch.double)

    def forward(self, u_now: torch.tensor, y_prev: torch.tensor):
        # x[k+1] = Ax[k] + Bu[k]
        y_now = self.B @ u_now.reshape(self.num_inputs)
        return y_now

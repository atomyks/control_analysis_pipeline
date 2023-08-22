import torch
import torch.nn as nn
from control_analysis_pipeline.model.model import Model


class BaseLinearModel(Model):
    """
    Similar to https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html
    """

    def __init__(self, num_actions=1, num_states=1):
        super(BaseLinearModel, self).__init__(num_action=num_actions, num_states=num_states)

        self.A = nn.Linear(num_states, num_states, bias=False, dtype=torch.double)
        self.B = nn.Linear(num_actions, num_states, bias=False, dtype=torch.double)

    def forward(self, a_input: torch.tensor, y_last: torch.tensor):
        '''
        Linear model forward pass y[k] = A*y[k-1] + B*u[k].
        :param a_input: torch.tensor, BATCH x NUM_INPUTS, system action
        :param y_last: torch.tensor, BATCH x NUM_STATES, system state
        :return:                
        '''
        y_now = self.A(y_last) + self.B(a_input)
        return y_now
    
    def set_model_matrices(self, A=None, B=None):
        '''
        Set the model matrices A and B. Be aware that this will create new parameters for the model and will reset requires_grad to True.
        :param A: torch.tensor, NUM_STATES x NUM_STATES, model matrix A
        :param B: torch.tensor, NUM_STATES x NUM_INPUTS, model matrix B
        '''
        if A is not None:
            self.A.weight = nn.parameter.Parameter(A)
        if B is not None:
            self.B.weight = nn.parameter.Parameter(B)

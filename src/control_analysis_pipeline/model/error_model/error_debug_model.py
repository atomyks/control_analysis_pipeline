import torch
import torch.nn as nn
from control_analysis_pipeline.model.error_model.error_model import ErrorModel
from control_analysis_pipeline.utils.normalizer import TorchNormalizer
import gpytorch
import gc
from typing import Optional


class ErrorDebugModel(ErrorModel):
    """
    Similar to https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html
    """

    def __init__(self, dt: float = 0.1):
        num_actions=1
        num_states=1
        num_errors=1
        action_history_size=1
        state_history_size=1
        super(ErrorDebugModel, self).__init__(num_actions=num_actions, num_states=num_states, num_errors=num_errors,
                                           action_history_size=action_history_size,
                                           state_history_size=state_history_size,
                                           sig_action_names=["STEER"], 
                                           sig_state_names=["STEER"])


    def forward(self,
                # regressors: Optional[torch.tensor] = None,
                u_input: Optional[torch.tensor] = None,
                y_last: Optional[torch.tensor] = None):
        """
        :param regressors: torch.tensor, BATCH x NUM_REGRESSORS, GP input
        :param u_input: torch.tensor, BATCH x NUM_INPUTS, system action
        :param y_last: torch.tensor, BATCH x NUM_STATES, system state
        :return:
                 output - gpytorch.distributions.MultitaskMultivariateNormal
                        - .mean (DATA_LENGTH x NUM_OUTPUTS)
                        - .stddev (DATA_LENGTH x NUM_OUTPUTS)
                        - .covariance_matrix (NUM_OUTPUTS * DATA_LENGTH x NUM_OUTPUTS * DATA_LENGTH)
        """

        output = 0.00
        mean = 0.0
        lower = -0.1
        upper = 0.1
        cov = None

        return [[output]], [[0.0]]  #, mean #, lower, upper, cov

    def init_learning(self):
        return None


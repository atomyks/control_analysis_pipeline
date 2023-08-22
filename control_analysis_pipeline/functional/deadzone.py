import torch
import torch.nn as nn
from control_analysis_pipeline.functional.base_functional import BaseNongradModel
from control_analysis_pipeline.functional.nongradient_parameter import NongradParameter


class Deadzone(BaseNongradModel):

    def __init__(self):
        super(Deadzone, self).__init__()

        self.r_lambd = NongradParameter(torch.zeros((1,)), lb=0., ub=20.1, precision=0.1)
        self.register_nongrad_parameter(name="r_lambd", value=self.r_lambd)

        self.l_lambd = NongradParameter(torch.zeros((1,)), lb=-20.1, ub=0., precision=0.01)
        self.register_nongrad_parameter(name="l_lambd", value=self.l_lambd)

    def forward(self, x: torch.tensor):
        """
        :param x: Calculates the deadzone of each element.
                  Parameters of deadzone are the same for each input element
        :return:
        """

        y = torch.where(x < self.l_lambd or self.r_lambd < x, x, 0.0)
        return y

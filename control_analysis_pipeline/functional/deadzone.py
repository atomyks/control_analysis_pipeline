import torch
import torch.nn as nn
from control_analysis_pipeline.model.model import Model
from control_analysis_pipeline.model.nongradient_parameter import NongradParameter


class Deadzone(Model):
    """
    Deadzone model. Inherits from Model class, but only uses the non-gradient parameters.
    """

    def __init__(self, r_lb=0.0, r_ub=20.0, r_precision=0.1,
                 l_lb=-20.0, l_ub=0.0, l_precision=0.1):
        super(Deadzone, self).__init__()

        self.r_lambd = NongradParameter(torch.zeros((1,)), lb=r_lb, ub=r_ub, precision=r_precision)
        self.register_nongrad_parameter(name="r_lambd", value=self.r_lambd)

        self.l_lambd = NongradParameter(torch.zeros((1,)), lb=l_lb, ub=l_ub, precision=l_precision)
        self.register_nongrad_parameter(name="l_lambd", value=self.l_lambd)

    def forward(self, x: torch.tensor):
        """
        :param x: Calculates the deadzone of each element.
                  Parameters of deadzone are the same for each input element
        :return:
        """
        y = torch.where(x < self.l_lambd or self.r_lambd < x, x, 0.0)
        # print(f"deadzone: x {x}  ->  y {y}  {x < self.l_lambd or self.r_lambd < x}, {self.l_lambd.get()}, ")

        return y

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

        self.sc = NongradParameter(torch.zeros((1,)), lb=1.0, ub=0.0, precision=0.1)
        self.register_nongrad_parameter(name="sc", value=self.sc)

    def forward(self, x: torch.tensor):
        """
        :param x: Calculates the deadzone of each element.
                  Parameters of deadzone are the same for each input element
        :return:
        """
        x1 = torch.where(torch.logical_or(x < self.l_lambd, self.r_lambd < x), x, 0.0)
        x2 = torch.where(self.r_lambd <= x1, x1 - self.r_lambd.get() * self.sc.get(), x1)
        y = torch.where(x2 <= self.l_lambd, x2 - self.l_lambd.get() * self.sc.get(), x2)
        return y


if __name__ == "__main__":
    m = Deadzone(r_lb=0.0, r_ub=1.0, r_precision=0.1,
                 l_lb=-1.0, l_ub=0.0, l_precision=0.1)
    m.r_lambd.set(torch.tensor([1.0]))
    m.l_lambd.set(torch.tensor([-1.0]))
    m.sc.set(torch.tensor([0.5]))

    in_ = torch.tensor([[2.0], [-2.0], [0.2], [0.0]])
    print(in_.shape)

    print("INPUT")
    print(in_)
    print()
    print("OUTPUT")
    print(m.forward(in_))

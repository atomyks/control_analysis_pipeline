import torch
import torch.nn as nn
from control_analysis_pipeline.model.model import Model
from control_analysis_pipeline.model.nongradient_parameter import NongradParameter


class Clip(Model):
    """
    Deadzone model. Inherits from Model class, but only uses the non-gradient parameters.
    """

    def __init__(self, symmetric=True,
                 upper_limit_ubound=1.0, lower_limit_ubound=0.0,
                 upper_limit_lbound=0.0, lower_limit_lbound=-1.0,
                 precision=0.1):
        super(Clip, self).__init__()
        self.symmetric = symmetric
        self.u_bound = NongradParameter(torch.zeros((1,)),
                                        lb=lower_limit_ubound,
                                        ub=upper_limit_ubound,
                                        precision=precision)
        self.register_nongrad_parameter(name="u_bound", value=self.u_bound)

        if not self.symmetric:
            self.l_bound = NongradParameter(torch.zeros((1,)),
                                            lb=lower_limit_lbound,
                                            ub=upper_limit_lbound,
                                            precision=precision)
            self.register_nongrad_parameter(name="l_bound", value=self.l_bound)

    def forward(self, x: torch.tensor):
        """
        :param x: Clip values of each element.
        :return:
        """

        if self.symmetric:
            tmp = torch.where(x > self.u_bound.get(), self.u_bound.get(), x)
            y = torch.where(tmp < -self.u_bound.get(), -self.u_bound.get(), tmp)
        else:
            tmp = torch.where(x > self.u_bound.get(), self.u_bound.get(), x)
            y = torch.where(tmp < self.l_bound.get(), self.l_bound.get(), tmp)

        return y


if __name__ == "__main__":
    pass

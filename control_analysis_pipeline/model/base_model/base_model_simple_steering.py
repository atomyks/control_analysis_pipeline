import torch
import torch.nn as nn
from control_analysis_pipeline.model.model import Model
from control_analysis_pipeline.functional.deadzone import Deadzone
from control_analysis_pipeline.model.nongradient_parameter import NongradParameter


class SimpleSteering(Model):
    def __init__(self):
        super(SimpleSteering, self).__init__(num_actions=1, num_states=2)
        self.batch_size = 1

        self.layer1 = Deadzone(r_lb=0.0, r_ub=0.02, r_precision=0.001,
                               l_lb=-0.02, l_ub=0.0, l_precision=0.001)
        self.register_model("layer1", self.layer1)

        self.time_const = NongradParameter(torch.zeros((1,)), lb=0.01, ub=0.2, precision=0.05)
        self.register_nongrad_parameter(name="time_constant", value=self.time_const)

        self.steer_rate_e = NongradParameter(torch.zeros((1,)), lb=0.0, ub=0.12, precision=0.001)
        self.register_nongrad_parameter(name="steer_rate_e", value=self.steer_rate_e)

        self.dt = 0.03

        self.last_steer_rate = 0.0

    def forward(self, a_input: torch.tensor, y_last: torch.tensor):
        '''
        :param a_input: torch.tensor, BATCH x NUM_INPUTS, system action
        :param y_last: torch.tensor, BATCH x NUM_STATES, system state
        :return:
        '''
        error = -(y_last - a_input)
        steer_rate = self.layer1(error + self.last_steer_rate) / self.time_const.get()

        steer_rate = min(max(steer_rate, -3.0), 3.0)  # limit for steering velocity
        y_output = y_last + steer_rate * self.dt  # Forward euler
        self.last_steer_rate = steer_rate * self.steer_rate_e.get()

        return y_output

    def reset(self):
        self.last_steer_rate = 0.0

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

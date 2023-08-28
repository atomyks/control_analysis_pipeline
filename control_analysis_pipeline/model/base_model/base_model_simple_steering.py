import torch
import torch.nn as nn
from control_analysis_pipeline.model.model import Model
from control_analysis_pipeline.functional.deadzone import Deadzone
from control_analysis_pipeline.model.nongradient_parameter import NongradParameter
from control_analysis_pipeline.model.delay_model.delay_model import InputDelayModel


class SimpleSteering(Model):
    def __init__(self, dt: float = 0.1):
        super(SimpleSteering, self).__init__(num_actions=1, num_states=1)
        self.batch_size = 1

        self.delay_layer = InputDelayModel(num_actions=1)
        self.delay_layer.reset()
        self.register_model("delay_layer", self.delay_layer)

        self.deadzone_layer = Deadzone(r_lb=0.0, r_ub=0.02, r_precision=0.001,
                                       l_lb=-0.02, l_ub=0.0, l_precision=0.001)
        self.register_model("deadzone_layer", self.deadzone_layer)

        self.time_const = NongradParameter(torch.zeros((1,)), lb=0.01, ub=0.2, precision=0.05)
        self.register_nongrad_parameter(name="time_constant", value=self.time_const)

        self.steer_rate_e = NongradParameter(torch.zeros((1,)), lb=0.0, ub=0.01, precision=0.001)
        self.register_nongrad_parameter(name="steer_rate_e", value=self.steer_rate_e)

        self.dt = dt

        self.last_steer_rate = 0.0

        self.loss_fn = nn.L1Loss()

    def forward(self, a_input: torch.tensor, y_last: torch.tensor):
        '''
        :param a_input: torch.tensor, BATCH x NUM_INPUTS, system action
        :param y_last: torch.tensor, BATCH x NUM_STATES, system state
        :return:
        '''

        a_input_delayed = self.delay_layer(a_input)

        error = -(y_last - a_input_delayed)
        steer_rate = self.deadzone_layer(error + self.last_steer_rate) / self.time_const.get()

        steer_rate = min(max(steer_rate, -3.0), 3.0)  # limit for steering velocity
        y_output = y_last + steer_rate * self.dt  # Forward euler
        self.last_steer_rate = steer_rate * self.steer_rate_e.get()

        return y_output

    def reset(self):
        self.last_steer_rate = 0.0
        self.delay_layer.reset()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

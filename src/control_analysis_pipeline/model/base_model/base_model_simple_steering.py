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

        self.deadzone_layer = Deadzone(r_lb=0.0019, r_ub=0.0020, r_precision=0.0001,
                                       l_lb=-0.0020, l_ub=-0.0019, l_precision=0.0001)
        self.register_model("deadzone_layer", self.deadzone_layer)

        self.time_const = NongradParameter(torch.zeros((1,)), lb=0.05, ub=0.52, precision=0.01)
        self.register_nongrad_parameter(name="time_constant", value=self.time_const)

        self.dt = dt

        self.loss_fn = nn.L1Loss()

    def forward(self, action: torch.tensor, state: torch.tensor):
        '''
        :param action: torch.tensor, BATCH x NUM_INPUTS, system action
        :param state: torch.tensor, BATCH x NUM_STATES, system state
        :return:
        '''

        action_delayed = self.delay_layer(action)

        error = -(state - action_delayed)
        steer_rate = self.deadzone_layer(error) / self.time_const.get()

        steer_rate = min(max(steer_rate, -3.0), 3.0)  # limit for steering velocity

        y_output = state + steer_rate * self.dt  # Forward euler

        return y_output, action_delayed

    def reset(self):
        self.delay_layer.reset()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
import torch
import torch.nn as nn
from control_analysis_pipeline.model.model import Model
from control_analysis_pipeline.functional.deadzone import Deadzone
from control_analysis_pipeline.model.nongradient_parameter import NongradParameter
from control_analysis_pipeline.model.delay_model.delay_model import InputDelayModel


class SimpleSteeringHist(Model):
    def __init__(self, dt: float = 0.1):
        super(SimpleSteeringHist, self).__init__(num_actions=1, num_states=1)
        self.batch_size = 1

        # Set delay layer
        self.delay_layer = InputDelayModel(num_actions=1)
        self.delay_layer.reset()
        self.register_model("delay_layer", self.delay_layer)

        # Set dead-zone layer
        self.deadzone_layer = Deadzone()
        self.deadzone_layer.sc.set(1.0)
        self.deadzone_layer.sc.trainable = False
        self.deadzone_layer.r_lambd.trainable = False
        self.deadzone_layer.l_lambd.trainable = False
        self.register_model("deadzone_layer", self.deadzone_layer)

        # Register additional model parameters
        self.time_const = NongradParameter(torch.zeros((1,)), lb=0.05, ub=0.5, precision=0.01)
        self.register_nongrad_parameter(name="time_constant", value=self.time_const)

        self.r_lambd_move = NongradParameter(torch.zeros((1,)), lb=0.001, ub=0.003, precision=0.0001)
        self.register_nongrad_parameter(name="r_lambd_move", value=self.r_lambd_move)

        self.l_lambd_move = NongradParameter(torch.zeros((1,)), lb=-0.003, ub=-0.001, precision=0.0001)
        self.register_nongrad_parameter(name="l_lambd_move", value=self.l_lambd_move)

        self.r_lambd_stop = NongradParameter(torch.zeros((1,)), lb=0.002, ub=0.004, precision=0.0001)
        self.register_nongrad_parameter(name="r_lambd_stop", value=self.r_lambd_stop)

        self.l_lambd_stop = NongradParameter(torch.zeros((1,)), lb=-0.004, ub=-0.002, precision=0.0001)
        self.register_nongrad_parameter(name="l_lambd_stop", value=self.l_lambd_stop)

        self.deadzone_layer.r_lambd.set(self.r_lambd_move.get())
        self.deadzone_layer.l_lambd.set(self.l_lambd_move.get())

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

        if (abs(steer_rate) < 0.00000001):
            self.deadzone_layer.r_lambd.set(self.r_lambd_stop.get())  # 0.0032
            self.deadzone_layer.l_lambd.set(self.l_lambd_stop.get())
        else:
            self.deadzone_layer.r_lambd.set(self.r_lambd_move.get())  # 0.0018
            self.deadzone_layer.l_lambd.set(self.l_lambd_move.get())

        return y_output, action_delayed

    def reset(self):
        self.delay_layer.reset()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

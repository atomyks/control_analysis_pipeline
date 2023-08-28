import torch
import torch.nn as nn
from control_analysis_pipeline.model.model import Model
from control_analysis_pipeline.functional.deadzone import Deadzone
from control_analysis_pipeline.model.nongradient_parameter import NongradParameter
from control_analysis_pipeline.model.delay_model.delay_model import InputDelayModel


class Steeringv2(Model):
    def __init__(self, dt: float = 0.1):
        super(Steeringv2, self).__init__(num_actions=1, num_states=2)
        self.batch_size = 1

        self.delay_layer = InputDelayModel(num_actions=1)
        self.delay_layer.reset()
        self.register_model("delay_layer", self.delay_layer)

        self.deadzone_layer = Deadzone(r_lb=0.0011, r_ub=0.0013, r_precision=0.0001,
                                       l_lb=-0.0012, l_ub=-0.0010, l_precision=0.0001)
        self.register_model("deadzone_layer", self.deadzone_layer)

        # self.steer_rate_e = NongradParameter(torch.zeros((1,)), lb=0.0, ub=1.0, precision=0.01)
        # self.register_nongrad_parameter(name="steer_rate_e", value=self.steer_rate_e)

        self.dt = dt

        self.loss_fn = nn.L1Loss()

        self.integral = 0.0
        self.last_error = 0.0
        self.last_steer_rate = 0.0

        self.Kp = NongradParameter(torch.zeros((1,)), lb=25.0, ub=26.0, precision=1.0)
        self.register_nongrad_parameter(name="Kp", value=self.Kp)

        self.Ki = NongradParameter(torch.zeros((1,)), lb=0.012, ub=0.013, precision=0.001)
        self.register_nongrad_parameter(name="Ki", value=self.Ki)

        self.Kd = NongradParameter(torch.zeros((1,)), lb=0.038, ub=0.039, precision=0.001)
        self.register_nongrad_parameter(name="Kd", value=self.Kd)

        self.J = NongradParameter(torch.zeros((1,)), lb=0.7, ub=0.8, precision=0.1)
        self.register_nongrad_parameter(name="J", value=self.J)

        self.b3 = NongradParameter(torch.zeros((1,)), lb=8.2, ub=8.3, precision=0.1)
        self.register_nongrad_parameter(name="b3", value=self.b3)

        self.T_f = NongradParameter(torch.zeros((1,)), lb=0.0, ub=0.001, precision=0.001)
        self.register_nongrad_parameter(name="T_f", value=self.T_f)

        self.max_vel = NongradParameter(torch.zeros((1,)), lb=99.0, ub=100.0, precision=1.0)
        self.register_nongrad_parameter(name="max_vel", value=self.max_vel)

        # self.max_vel = None

    def forward(self, a_input: torch.tensor, y_last: torch.tensor):
        '''
        :param a_input: torch.tensor, BATCH x NUM_INPUTS, system action
        :param y_last: torch.tensor, BATCH x NUM_STATES, system state
        :return:
        '''

        # a_input (BATCH x 1)
        # y_last (BATCH x 2);  y_last[:, 0] = angle; y_last[:, 1] - steering speed

        # delay
        delayed_input = self.delay_layer(a_input)

        # Virtual PID regulator
        error = -(y_last[:, 0] - delayed_input)
        self.integral += error
        T_in = error * self.Kp.get() + self.integral * self.Ki.get() + (self.last_error - error) * self.Kd.get()
        self.last_error = error

        # Virtual system
        y_output = y_last + self.system_get_f(y_last, self.J.get(), T_in, self.b3.get(),
                                              self.T_f.get()) * self.dt  # Forward euler

        y_output[:, 1] = self.deadzone_layer(y_output[:, 1])
        # y_output[:, 1] = min(max(y_output[:, 1], -self.max_vel.get()), self.max_vel)

        # limit for steering velocity
        tmp = torch.where(y_output[:, 1] > self.max_vel.get(), self.max_vel.get(), y_output[:, 1])
        y_output[:, 1] = torch.where(tmp < -self.max_vel.get(), -self.max_vel.get(), tmp)

        # self.last_steer_rate = y_output[:, 1] * self.steer_rate_e.get()

        return y_output

    def system_get_f(self, x, J, T_in, b3, T_f):
        # x (BATCH x 2);  x[:, 0] = angle; x[:, 1] - steering speed
        # x1 = x[:, 0]
        x2 = x[:, 1]
        x1_dot = x2
        x2_dot = 1.0 / J * (T_in - b3 * x2 - T_f)

        x_dot = torch.cat((x1_dot.reshape((self.batch_size, 1)), x2_dot.reshape((self.batch_size, 1))), dim=-1)
        return x_dot

    def reset(self):
        self.last_steer_rate = 0.0
        self.last_error = 0.0
        self.integral = 0.0
        self.delay_layer.reset()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

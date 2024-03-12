import torch
import torch.nn as nn
from control_analysis_pipeline.model.model import Model
from control_analysis_pipeline.functional.deadzone import Deadzone
from control_analysis_pipeline.model.nongradient_parameter import NongradParameter
from control_analysis_pipeline.model.delay_model.delay_model import InputDelayModel


class Drive_V1(Model):
    """
    Simple first order drive model with steady state error. 
    """
    def __init__(self, dt: float = 0.1):
        super(Drive_V1, self).__init__(num_actions=1, num_states=1, 
                                                 sig_action_names=["VX_DES"], 
                                                 sig_state_names=["VX"])
        self.batch_size = 1

        # Set delay layer
        self.delay_layer = InputDelayModel(num_actions=1)
        self.delay_layer.reset()
        self.register_model("delay_layer", self.delay_layer)

        # Register additional model parameters
        self.time_const = NongradParameter(torch.zeros((1,)), lb=0.0, ub=2.0, precision=0.02)
        self.register_nongrad_parameter(name="time_constant", value=self.time_const)

        self.dt = NongradParameter(torch.zeros((1,)), trainable=False)
        self.register_nongrad_parameter(name="dt", value=self.dt)

        self.max_acceleration = NongradParameter(torch.zeros((1,)), lb=0.1, ub=3.0, precision=0.1, trainable=True)
        self.register_nongrad_parameter(name="max_acceleration", value=self.max_acceleration)

        self.input_scale = NongradParameter(torch.zeros((1,)), lb=0.9, ub=1.0, precision=0.005, trainable=True)
        self.register_nongrad_parameter(name="input_scale", value=self.input_scale)

        self.time_const.set(0.1)
        self.dt.set(dt)

        self.loss_fn = nn.L1Loss()


    def forward(self, action: torch.tensor, state: torch.tensor):
        '''
        :param action: torch.tensor, BATCH x NUM_INPUTS, system action
        :param state: torch.tensor, BATCH x NUM_STATES, system state
        :return:
        '''

        action = action.reshape((self.batch_size, self.num_actions))
        state = state.reshape((self.batch_size, self.num_states))

        # Simulate the delay of the input signal
        action_delayed = self.delay_layer(action) * self.input_scale.get()

        # Compute the error between the required velocity and current velocity
        error = -(state - action_delayed)  # try multiplying action by number smaller than 1

        # Compute acceleration
        acceleration = error / self.time_const.get()

        # Limit acceleration
        acceleration = torch.clamp(acceleration, min=-10.0, max=self.max_acceleration.get())

        # Integrate the system
        next_state = state + acceleration * self.dt.get() # Forward euler

        return next_state, action_delayed

    def system_equation(self, action):
        pass

    def reset(self): 
        self.delay_layer.reset()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

import torch
import torch.nn as nn
from control_analysis_pipeline.model.model import Model
from control_analysis_pipeline.functional.deadzone import Deadzone
from control_analysis_pipeline.model.nongradient_parameter import NongradParameter
from control_analysis_pipeline.model.delay_model.delay_model import InputDelayModel


class SimpleVelocity(Model):
    def __init__(self, dt: float = 0.1):
        super(SimpleVelocity, self).__init__(num_actions=1, num_states=1, 
                                                 sig_action_names=["VX_DES"], 
                                                 sig_state_names=["VX"])
        self.batch_size = 1

        # Set delay layer
        self.delay_layer = InputDelayModel(num_actions=1)
        self.delay_layer.reset()
        self.register_model("delay_layer", self.delay_layer)

        # Register additional model parameters
        self.time_const = NongradParameter(torch.zeros((1,)), lb=0.05, ub=0.5, precision=0.01)
        self.register_nongrad_parameter(name="time_constant", value=self.time_const)

        self.dt = NongradParameter(torch.zeros((1,)), trainable=False)
        self.register_nongrad_parameter(name="dt", value=self.dt)
        
        self.time_const.set(0.1)
        self.dt.set(dt)
        

    def forward(self, action: torch.tensor, state: torch.tensor):
        '''
        :param action: torch.tensor, BATCH x NUM_INPUTS, system action
        :param state: torch.tensor, BATCH x NUM_STATES, system state
        :return:
        '''

        action = action.reshape((self.batch_size, self.num_actions))
        state = state.reshape((self.batch_size, self.num_states))

        # Simulate the delay of the input signal
        action_delayed = self.delay_layer(action)

        # Compute the error between the required velocity and current velocity
        error = -(state - action_delayed)

        # Compute steering rate (First order system)
        acceleration = error / self.time_const.get()

        # Use some saturation for the steering rate
        acceleration = torch.clamp(acceleration, min=-1.0, max=1.0)  # limit for steering velocity

        # Integrate the system
        next_state = state + acceleration * self.dt.get() # Forward euler

        return next_state, action_delayed

    def reset(self): 
        self.delay_layer.reset()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

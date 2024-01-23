import torch
import torch.nn as nn
from control_analysis_pipeline.model.model import Model
from control_analysis_pipeline.functional.deadzone import Deadzone
from control_analysis_pipeline.model.nongradient_parameter import NongradParameter
from control_analysis_pipeline.model.delay_model.delay_model import InputDelayModel


class KinematicBicycleSteerVel(Model):
    def __init__(self, dt: float = 0.1):
        super(KinematicBicycleSteerVel, self).__init__(num_actions=2, num_states=3, 
                                                 sig_action_names=["STEER", "VX"], 
                                                 sig_state_names=["POS_X", "POS_Y", "YAW"])
        self.batch_size = 1

        # Register additional model parameters

        self.wheelbase = NongradParameter(torch.zeros((1,)), trainable=False)
        self.register_nongrad_parameter(name="wheelbase", value=self.wheelbase)
        self.wheelbase.set(2.2)

        self.dt = NongradParameter(torch.zeros((1,)), trainable=False)
        self.register_nongrad_parameter(name="dt", value=self.dt)
        self.dt.set(0.02)
        

    def forward(self, action: torch.tensor, state: torch.tensor):
        '''
        :param action: torch.tensor, BATCH x NUM_INPUTS, system action
        :param state: torch.tensor, BATCH x NUM_STATES, system state
        :return:
        '''

        action = action.reshape((self.batch_size, self.num_actions))
        state = state.reshape((self.batch_size, self.num_states))

        # actions
        steer = action[:, 0]
        v_x = action[:, 1]
        
        # states
        yaw = state[:, 2]

        # model equations
        f_x = torch.zeros_like(state)
        f_x[:, 0] = v_x * torch.cos(yaw)
        f_x[:, 1] = v_x * torch.sin(yaw)
        f_x[:, 2] = v_x * torch.tan(steer) / self.wheelbase.get()

        # Integrate the model
        next_state = state + f_x * self.dt.get() # Forward euler

        return next_state, action


    def set_batch_size(self, batch_size):
        self.batch_size = batch_size


if __name__=="__main__":
    model = KinematicBicycleSteerVel()
    model(torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0, 0.0]))
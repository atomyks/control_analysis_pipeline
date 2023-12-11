import torch
import torch.nn as nn
from control_analysis_pipeline.model.model import Model
from control_analysis_pipeline.functional.deadzone import Deadzone
from control_analysis_pipeline.model.nongradient_parameter import NongradParameter
from control_analysis_pipeline.model.delay_model.delay_model import InputDelayModel


class SimpleSteeringHyst(Model):
    def __init__(self, dt: float = 0.1):
        super(SimpleSteeringHyst, self).__init__(num_actions=1, num_states=1)
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

        self.dt = NongradParameter(torch.zeros((1,)), trainable=False)
        self.register_nongrad_parameter(name="dt", value=self.dt)
        self.dt.set(dt)
        

    def forward(self, action: torch.tensor or list, state: torch.tensor or list):
        '''
        :param action: torch.tensor, BATCH x NUM_INPUTS, system action
        :param state: torch.tensor, BATCH x NUM_STATES, system state
        :return:
        '''

        type_list = False
        if isinstance(action, list):
            type_list = True
            action = torch.tensor(action).reshape((self.batch_size, self.num_actions))
            state = torch.tensor(state).reshape((self.batch_size, self.num_states))

        # Simulate the delay of the input signal
        action_delayed = self.delay_layer(action)
        
        # Compute the error between the required steering angle and current steering angle
        error = -(state - action_delayed)

        # Compute steering rate (First order system)
        steer_rate = self.deadzone_layer(error) / self.time_const.get()

        # Use some saturation for the steering rate
        steer_rate = torch.clamp(steer_rate, min=-3.0, max=3.0)  # limit for steering velocity

        # Integrate the system
        next_state = state + steer_rate * self.dt.get() # Forward euler

        # Compute the deadzone hysteresis
        if (abs(steer_rate) < 0.00000001): # If in the "dead zone"
            self.deadzone_layer.r_lambd.set(self.r_lambd_stop.get())
            self.deadzone_layer.l_lambd.set(self.l_lambd_stop.get())
        else:  # If outside of the "dead zone"
            self.deadzone_layer.r_lambd.set(self.r_lambd_move.get())
            self.deadzone_layer.l_lambd.set(self.l_lambd_move.get())

        if type_list:
            return next_state.tolist(), action_delayed.tolist()
        return next_state, action_delayed

    def reset(self): 
        self.delay_layer.reset()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

if __name__=="__main__":
    model = SimpleSteeringHyst()

    model.load_params("./base_model_save")

    a = torch.tensor([[1.0]])
    s = torch.tensor([[2.0]])
    print(model(a, s))
    print(model.forward(a, s))

    print("----------")

    print(f"Initial state: {s}")

    s = [2.0]
    a = [1.0]

    for t in range(5):
        s, aa_ = model(a, s)
        print(f"Input: {a}   State: {s}   (t): {t}")
        


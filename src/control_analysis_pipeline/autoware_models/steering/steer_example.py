from control_analysis_pipeline.model.base_model.base_model_simple_steering import SimpleSteering
import torch

class SteerExample:

    def __init__(self) -> None:
        self.steer_model = SimpleSteering()
        # model parameters
        self.steer_model.delay_layer(torch.tensor(5))
        self.steer_model.deadzone_layer.r_lambd.set(torch.tensor(0.0))
        self.steer_model.deadzone_layer.l_lambd.set(torch.tensor(0.0))
        self.steer_model.deadzone_layer.sc.set(torch.tensor(0.0))
        self.steer_model.time_const.set(torch.tensor(0.05))
        self.steer_model.reset()

    def forward(self, action, state):  # Required
        """
        Calculate forward pass through the model and returns next_state.
        """
        next_state, action = self.steer_model(torch.tensor(action), torch.tensor(state))
        return next_state[0].tolist()  # next state

    def get_state_names(self):  # Required
        """
        Return list of string names of the model states (outputs).
        """
        return self.steer_model.get_state_names()

    def get_action_names(self):  # Required
        """
        Return list of string names of the model actions (inputs).
        """
        return self.steer_model.get_action_names()

    def reset(self):  # Required
        """
        Reset model. This function is called after load_params().
        """
        pass

    def load_params(self, path):  # Optional
        """
        Load parameters of the model.
        Inputs:
            - path: Path to a parameter file to load by the model.
        """
        pass

    def dtSet(self, dt):
        """
        Set dt of the model.
        Inputs:
            - dt: time step
        """
        self.steer_model.dt = dt

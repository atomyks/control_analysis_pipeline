from control_analysis_pipeline.model.base_model.base_model_simple_velocity import SimpleVelocity
import torch

class DriveExample:

    def __init__(self) -> None:
        self.drive_model = SimpleVelocity()
        self.drive_model.delay_layer.delay_parm.set(4)
        self.drive_model.time_const.set(0.1)
        self.drive_model.reset()

    def forward(self, action, state):  # Required
        """
        Calculate forward pass through the model and returns next_state.
        """
        next_state, action = self.drive_model(torch.tensor(action), torch.tensor(state))
        return next_state[0].tolist()  # next state
    
    def get_state_names(self):  # Required
        """
        Return list of string names of the model states (outputs).
        """
        return self.drive_model.get_state_names()

    def get_action_names(self):  # Required
        """
        Return list of string names of the model actions (inputs).
        """
        return self.drive_model.get_action_names()

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
        self.drive_model.dt.set(dt)
from control_analysis_pipeline.model.base_model.base_model_simple_steering_hysteresis import SimpleSteeringHyst
from control_analysis_pipeline.model.error_model.error_debug_model import ErrorDebugModel
import torch

class BaseError:

    def __init__(self) -> None:
        self.base = SimpleSteeringHyst()
        self.error = ErrorDebugModel()  # Currently dummy model

    def forward(self, action, state):  # Required
        """
        Calculate forward pass through the model and returns next_state.
        """
        next_state, delayed_action = self.base(torch.tensor(action), torch.tensor(state))
        next_state_error, _ = self.error(delayed_action, state)
        next_state_corrected = next_state + next_state_error
        return next_state_corrected[0].tolist()  # next state

    def get_state_names(self):  # Required
        """
        Return list of string names of the model states (outputs).
        """
        return self.base.get_state_names()

    def get_action_names(self):  # Required
        """
        Return list of string names of the model actions (inputs).
        """
        return self.base.get_action_names()

    def reset(self):  # Required
        """
        Reset model. This function is called after load_params().
        """
        self.base.reset()

    def load_params(self, path):  # Optional
        """
        Load parameters of the model. 
        Inputs:
            - path: Path to a parameter file to load by the model.
        """
        self.base.load_params(path)

    def dtSet(self, dt):
        """
        Set dt of the model.
        Inputs:
            - dt: time step
        """
        self.base.dt.set(dt)
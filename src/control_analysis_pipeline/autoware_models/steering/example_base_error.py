from control_analysis_pipeline.model.base_model.base_model_simple_steering_hysteresis import SimpleSteeringHyst
from control_analysis_pipeline.model.error_model.error_debug_model import ErrorDebugModel

class BaseError:

    def __init__(self) -> None:
        self.base = SimpleSteeringHyst()
        self.error = ErrorDebugModel()

    def forward(self, action, state):  # Required
        """
        Calculate forward pass through the model and returns next_state.
        """
        next_state, delayed_action = self.base(action, state)
        next_state_error, _ = self.error(delayed_action, state)
        return [next_state[0][0] + next_state_error[0][0]]  # next state
    
    def get_state_names(self):  # Required
        """
        Return list of string names of the model states (outputs).
        """
        return self.base.get_sig_state_names()

    def get_action_names(self):  # Required
        """
        Return list of string names of the model actions (inputs).
        """
        return self.base.get_sig_action_names()

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
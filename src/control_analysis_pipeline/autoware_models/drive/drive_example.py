from control_analysis_pipeline.model.base_model.base_model_simple_velocity import SimpleVelocity

class DriveExample:

    def __init__(self) -> None:
        self.drive_model = SimpleVelocity()

    def forward(self, action, state):  # Required
        """
        Calculate forward pass through the model and returns next_state.
        """
        next_state, action = self.drive_model(action, state)
        return next_state[0]  # next state
    
    def get_state_names(self):  # Required
        """
        Return list of string names of the model states (outputs).
        """
        return self.drive_model.get_sig_state_names()

    def get_action_names(self):  # Required
        """
        Return list of string names of the model actions (inputs).
        """
        return self.drive_model.get_sig_action_names()

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
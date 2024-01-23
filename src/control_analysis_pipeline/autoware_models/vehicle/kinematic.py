from control_analysis_pipeline.model.base_model.kinematic_bicycle_steer_vel import KinematicBicycleSteerVel

class KinematicModel:

    def __init__(self) -> None:
        self.kinematic_model = KinematicBicycleSteerVel()

    def forward(self, action, state):  # Required
        """
        Calculate forward pass through the model and returns next_state.
        """
        next_state, action = self.kinematic_model(action, state)
        return next_state[0]  # next state
    
    def get_state_names(self):  # Required
        """
        Return list of string names of the model states (outputs).
        """
        return self.kinematic_model.get_sig_state_names()

    def get_action_names(self):  # Required
        """
        Return list of string names of the model actions (inputs).
        """
        return self.kinematic_model.get_sig_action_names()

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
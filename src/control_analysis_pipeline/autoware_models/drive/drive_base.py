from control_analysis_pipeline.model.base_model.base_model_drive_v1 import Drive_V1
import torch

class DriveBase:

    def __init__(self) -> None:
        self.drive_model = Drive_V1()
        # model parameters
        # self.drive_model.time_const.set(0.74)
        # self.drive_model.max_acceleration.set(2.7)
        # self.drive_model.input_scale.set(0.98)
        # self.drive_model.delay_layer.delay_parm.set(1)
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
        self.drive_model.reset()

    def load_params(self, path):  # Optional
        """
        Load parameters of the model.
        Inputs:
            - path: Path to a parameter file to load by the model.
        """
        print(path)
        self.drive_model.load_params(path)

    def dtSet(self, dt):
        """
        Set dt of the model.
        Inputs:
            - dt: time step
        """
        self.drive_model.dt.set(dt)


if __name__=="__main__":
    model = DriveBase()
    print(model.get_action_names())

    model.load_params("$HOME/autoware_model_params/drive_base_parameters.pt")

import torch
import torch.nn as nn
from control_analysis_pipeline.regressor.regressor_factory import RegressorFactory

class Model(nn.Module):
    """
    Base class for all models. All models should inherit from this class.
    """

    def __init__(self, num_action=1, num_states=1, action_history_size=1, state_history_size=1):
        """
        :param num_action: int, number of inputs to the model
        :param num_states: int, number of outputs from the model
        :param action_history_size: int, number of previous actions to use as input
        :param state_history_size: int, number of previous states to use as input
        """

        super(Model, self).__init__()

        # Learning toggles, dictates what optimizer is used to train the model
        self.back_prop = False
        self.nongrad_params = None

        # Basic model parameters
        self.num_action = num_action
        self.num_states = num_states

        # Regressor parameters
        self.action_history_size = action_history_size
        self.state_history_size = state_history_size
        self.reg = RegressorFactory(batch_size=1, num_actions=self.num_actions,
                                    num_states=self.num_states, action_history_size=self.action_history_size,
                                    state_history_size=self.state_history_size)

    def forward(self,
                regressors: torch.tensor or None = None,
                a_input: torch.tensor or None = None,
                y_last: torch.tensor or None = None):
        """
        By default, the model simply returns the regressors. Override this function if you need to do some computation
        :param regressors: torch.tensor, BATCH x NUM_REGRESSORS, model regressors
        :param a_input: torch.tensor, BATCH x NUM_INPUTS, system action
        :param y_last: torch.tensor, BATCH x NUM_STATES, system state
        :return:
        """
        return regressors

    def get_regressors(self, u_input: torch.tensor, y_last: torch.tensor):
        """
        :param u_input: torch.tensor, 1 x INPUTS, system input
        :param y_last: torch.tensor, 1 x STATES, system input
        :return: computed regressors (1 x NUM_REGRESSORS)
        """
        regressors = self.reg(u_input,  # torch.rand((batch_size, num_actions))
                              y_last)  # torch.rand((batch_size, num_states))
        return regressors

    def regressor_size(self):
        """
        Get the number of regressors

        :return: int, number of regressors
        """

        return len(self.reg)

    def init_learning(self):
        '''
        Used to initialize the learning process. This function is called before the first call of the learn function.
        Default implementation does nothing. Override this function if you need to do some initialization.
        :return:
        '''
        pass

    def init_regressor_history(self, 
                               batch_size: int = 1,
                               history_a: torch.tensor or None = None, 
                               history_s: torch.tensor or None = None):
        '''
        Initializes the history of the regressor. This function is called before each learning iteration.
        :param batch_size: Batch size of the regressor
        :param history_a: History of the action
        :param history_s: History of the state
        :return:
        '''
        # set batch size
        self.reg.set_batch_size(batch_size)

        # set history, defaults to zero if not set
        if history_a is None:
            history_a = torch.zeros((batch_size, self.action_history_size, self.num_inputs))
        if history_s is None:
            history_s = torch.zeros((batch_size, self.state_history_size, self.num_outputs))

        self.reg.set_history(history_a, history_s)
        
    def __str__(self):
        '''
        String operator overload for printing the model to the console.
        :return: String representation of the model
        '''
        str_out = str(self.reg)
        str_out += "\n"
        str_out += super(Model, self).__str__()
        return str_out

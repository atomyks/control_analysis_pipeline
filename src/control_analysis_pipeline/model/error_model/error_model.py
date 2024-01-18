import torch
from control_analysis_pipeline.model.model import Model


class ErrorModel(Model):
    """
    Base class for error models. Inherits from Model class, and has unique forward method.
    Error model takes action and state output of the base model and estimates prediction error of base model.

    action  ###############
    ------->#             #  error
    state   # error model #-------->
    ------->#             #
            ###############

    """

    def __init__(self, num_actions=1, num_states=1, num_errors=1, action_history_size=1, state_history_size=1, sig_action_names=None, sig_state_names=None):
        super(ErrorModel, self).__init__(num_actions=num_actions, num_states=num_states,
                                         action_history_size=action_history_size, state_history_size=state_history_size, 
                                         sig_action_names=sig_action_names, sig_state_names=sig_state_names)
        self.model_input = None
        self.model_output = None
        self.num_errors = num_errors

    def set_training_data(self,
                          train_s: torch.tensor or list,
                          train_a: torch.tensor or list,
                          train_e: torch.tensor or list):
        """

        Note: BATCH dimension can be either list or torch.tensor. DATA and last  dimension needs to be torch.tensor.
        :param train_s: States array. (BATCH x DATA_LENGTH x NUM_STATES)
        :param train_a: Action array. (BATCH x DATA_LENGTH x NUM_ACTIONS)
        :param train_e: Error array. (BATCH x DATA_LENGTH x NUM_OBSERVED_STATES)
        :return:
        """
        if type(train_s) == list:
            batch_dim = len(train_s)
            if not (len(train_s) == len(train_a) == len(train_e)):
                raise ValueError('dimension mismatch')
        else:
            if train_s.dim() == 2:
                batch_dim = 1
                train_s = train_s.reshape((batch_dim, train_s.shape[0], train_s.shape[1]))
                train_a = train_a.reshape((batch_dim, train_a.shape[0], train_a.shape[1]))
                train_e = train_e.reshape((batch_dim, train_e.shape[0], train_e.shape[1]))
            if train_s.dim() == 3:
                batch_dim, _, _ = train_s.shape
            else:
                raise ValueError('dimension mismatch')

        # data needs to be synchronized at all time -> we need to take the same history size from both signals
        required_history = max(self.action_history_size, self.state_history_size)

        idx = 0

        self.model_input = None
        self.model_output = None

        # needs to be done iteratively to support list on the input
        for i in range(batch_dim):
            # get dimensions of the current data signal
            train_s_single = train_s[i]
            data_length_s, num_states = train_s_single.shape
            train_a_single = train_a[i]
            data_length_a, num_actions = train_a_single.shape
            train_e_single = train_e[i]
            data_length_out, num_observed_states = train_e_single.shape

            # check dimensions of the current data signal
            if not (data_length_s == data_length_a == data_length_out):
                raise ValueError('dimension mismatch')
            if (not num_states == self.num_states) or (not num_actions == self.num_actions):
                raise ValueError('dimension mismatch')

            data_length = data_length_s  # since all of them should be the same it does not matter which one we take

            if self.model_input is None:
                self.model_input = torch.zeros((data_length - required_history, self.regressor_size()))
            else:
                self.model_input = torch.cat(
                    (self.model_input, torch.zeros((data_length - required_history, self.regressor_size()))
                     ), dim=0)
            if self.model_output is None:
                self.model_output = torch.zeros((data_length - required_history, num_observed_states))
            else:
                self.model_output = torch.cat(
                    (self.model_output, torch.zeros((data_length - required_history, num_observed_states))
                     ), dim=0)

            # set start and end of the history correctly for all signals
            s_history_start = required_history - self.state_history_size
            a_history_start = required_history - self.action_history_size

            # set regressor history
            history_s = train_s_single[s_history_start:required_history, :].reshape((1,
                                                                                     self.state_history_size,
                                                                                     self.num_states))
            history_a = train_a_single[a_history_start:required_history, :].reshape((1,
                                                                                     self.action_history_size,
                                                                                     self.num_actions))
            self.reg.set_history(history_a, history_s)

            # process training data
            for j in range(required_history, data_length):
                regressor = self.get_regressors(u_input=train_a_single[j, :], y_last=train_s_single[j, :])
                self.model_input[idx, :] = regressor
                self.model_output[idx, :] = train_e_single[j, :]
                idx += 1

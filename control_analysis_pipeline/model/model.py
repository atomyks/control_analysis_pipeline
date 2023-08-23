import torch
import torch.nn as nn
from control_analysis_pipeline.regressor.regressor_factory import RegressorFactory

class Model(nn.Module):
    """
    Base class for all models. All models should inherit from this class.
    """

    def __init__(self, num_actions=1, num_states=1, action_history_size=1, state_history_size=1):
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
        self.num_actions = num_actions
        self.num_states = num_states

        # Regressor parameters
        self.action_history_size = action_history_size
        self.state_history_size = state_history_size
        self.reg = RegressorFactory(batch_size=1, num_actions=self.num_actions,
                                    num_states=self.num_states, action_history_size=self.action_history_size,
                                    state_history_size=self.state_history_size)
        
        # Gradient loss function
        self.loss_fn = None

        # Non-gradient parameters
        self._nongrad_params = {}
        self._nongrad_params_flat = {}
        self._models = {}

    def enable_grad_learning(self, loss_function):
        """
        Enable gradient learning. This will enable backpropagation and set the loss function to the given loss function.
        :param loss_function: loss function to use for gradient learning
        """
        self.back_prop = True
        self.loss_fn = loss_function
        
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

    def init_learning(self):
        '''
        Used to initialize the learning process. This function is called before the first call of the learn function.
        Default implementation does nothing. Override this function if you need to do some initialization.
        :return:
        '''
        pass

    def register_nongrad_parameter(self, name, value):
        '''
        Registers a non-gradient parameter. This is optimized using a gradient-free optimizer.
        :param name: Name of the parameter
        :param value: Parameter to be added.
        :return:
        '''
        self._nongrad_params[name] = value

    def register_model(self, name, value):
        '''
        Registers a model. This allows for layered models.
        :param name: Name of the model
        :param value: Model to be added
        :return:
        '''
        self._models[name] = value

    def gen_nongrad_params_flat(self):
        '''
        Generates a flattened version of the non-gradient parameters by recursively flattening the submodels.
        :return: Flattened dictionary of non-gradient parameters
        '''

        self._nongrad_params_flat = self._nongrad_params
        for model_key in list(self._models.keys()):
            submodel_nongrad_params = self._models[model_key].gen_nongrad_params_flat()
            for submodel_key in list(submodel_nongrad_params.keys()):
                flat_key = model_key + "." + submodel_key
                self._nongrad_params_flat[flat_key] = submodel_nongrad_params[submodel_key]
        return self._nongrad_params_flat

    def get_search_space(self):
        '''
        Generates the search space for the model. This is used by the gradient-free optimizer to search for the optimal parameters.
        :return: Search space and the parameters
        '''

        search_space = {}
        optim_params = self.gen_nongrad_params_flat()
        for name in list(optim_params.keys()):
            search_space[name] = torch.arange(optim_params[name].lb,
                                              optim_params[name].ub,
                                              optim_params[name].precision)
        return search_space, optim_params

    def __str__(self):
        '''
        String operator overload for printing the model to the console.
        :return: String representation of the model
        '''
        str_out = str(self.reg)
        str_out += "\n"
        str_out += super(Model, self).__str__()
        str_out += "\n----------------------------------\n"
        str_out += f"Nongrad params: {self._nongrad_params.keys()}\n"
        str_out += f"Registered models: {self._models.keys()}\n"
        return str_out

    def __repr__(self):
        report = "\n----------------------------------\n"
        report += f"Nongrad params: {self._nongrad_params.keys()}\n"
        report += f"Registered models: {self._models.keys()}\n"
        return report
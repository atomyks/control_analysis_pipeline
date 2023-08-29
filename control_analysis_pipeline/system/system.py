from control_analysis_pipeline.model.base_model.base_model_linear import BaseLinearModel
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os
import gradient_free_optimizers as gfo
import copy


# Enum for model selection
class ModelType:
    BASE = 0
    ERROR = 1


class System:
    def __init__(self, loaded_data: dict = None, num_states: int = 1, num_actions: int = 1,
                 sampling_period: float = 0.01):
        if loaded_data is not None:
            self.loaded_data = loaded_data["data"]
        else:
            self.loaded_data = None

        self.training_data = None
        self.testing_data = None

        if loaded_data is not None:
            self.sampling_period = loaded_data["header"]["sampling_period"]
        else:
            self.sampling_period = sampling_period

        self.output_data = None

        # system delay
        self.base_model = BaseLinearModel(num_actions=num_actions, num_states=num_states)
        self.error_model = None
        self.inputs = None
        self.outputs = None

    def set_linear_model_matrices(self, A=None, B=None):
        self.base_model.set_model_matrices(A, B)

    def parse_config(self, config):
        self.inputs = []
        self.outputs = []

        # self.valid_data_enabled = config["valid_data_enabled"]

        for key in list(config["data_to_load"].keys()):
            if config["data_to_load"][key]["type"] == "input":
                self.inputs.append(key)
            if config["data_to_load"][key]["type"] == "output":
                self.outputs.append(key)

    def system_step(self, u_input: torch.tensor, state_now: torch.tensor,
                    sim_base_model=True, sim_error_model=True) -> (torch.tensor, torch.tensor):

        # if we are not simulating the base model, then the base model is just the identity
        state_next = state_now
        u_input_delayed = u_input
        if sim_base_model:
            state_next, u_input_delayed = self.base_model(u_input, state_now)

        error = torch.zeros((state_now.shape[0], self.base_model.num_states))
        if sim_error_model and self.error_model is not None:
            _, error, lower, upper = self.error_model(u_input=u_input_delayed, y_last=state_now)

        # state_next = state_next + error
        return state_next, u_input_delayed, error

    def simulate(self, input_array: torch.tensor, initial_state: torch.tensor = None,
                 use_base_model=True, use_error_model=True):
        """
        :param input_array: torch.tensor, (BATCH x TIME x NUM_INPUTS) or (TIME x NUM_INPUTS)
        :param initial_state: torch.tensor, (BATCH x NUM_INPUTS)
        :param use_base_model:
        :param use_error_model:
        :return:
        """

        if input_array.dim() == 3:
            batch_size, time_length, num_actions = input_array.shape
        elif input_array.dim() == 2:
            time_length, num_actions = input_array.shape
            batch_size = 1
        else:
            raise ValueError("Input array must be of shape (BATCH x TIME x NUM_INPUTS) or (TIME x NUM_INPUTS)")

        if initial_state is not None:
            state = initial_state.reshape((batch_size, self.base_model.num_states))
        else:
            state = torch.zeros((batch_size, self.base_model.num_states))

        action_delayed_array = torch.zeros((batch_size, time_length, num_actions))
        state_array = torch.zeros((batch_size, time_length, self.base_model.num_states))  # BATCH x TIME x STATES
        error_array = torch.zeros((batch_size, time_length, self.base_model.num_states))

        for t in range(time_length):
            # One-slice t:t+1 allows us to use the same code for batched and unbatched data while preserving correct dimensions
            state, action_delayed, error = self.system_step(input_array[..., t, :], state, use_base_model,
                                                            use_error_model)

            error_correct_dim = torch.cat((
                error,
                torch.zeros((state.shape[0], state.shape[1] - error.shape[1]))
            ), dim=-1)
            state_array[..., t, :] = state + error_correct_dim
            action_delayed_array[..., t, :] = action_delayed
            error_array[..., t, :] = error_correct_dim  # TODO needs to be solve

        return state_array, action_delayed_array, error_array

    def plot_simulation(self, input_array: torch.tensor, initial_state: torch.tensor = None,
                        true_state: torch.tensor = None,
                        ax: plt.Axes = None, show_hidden_states=True, show_input=True,
                        use_base_model=True, use_error_model=True):
        """
        :param input_array: torch.tensor, (TIME x NUM_INPUTS)
        :param initial_state: torch.tensor, (1 x NUM_STATES)
        :param true_state: torch.tensor, (TIME x NUM_STATES)
        :param use_base_model:
        :param use_error_model:
        :return:
        """

        state_array = None
        with torch.no_grad():
            # state_array (TIME x STATES)
            state_array, _, _ = self.simulate(input_array,
                                              initial_state=initial_state,
                                              use_base_model=use_base_model,
                                              use_error_model=use_error_model)

        state_array = state_array.numpy()

        if true_state is None:
            num_observed_states = 0
        else:
            _, num_observed_states = true_state.shape
        time_length, num_actions = input_array.shape

        if ax is None:
            fig, ax = plt.subplots()

        if show_input:
            for i in range(num_actions):
                label = f"Input: {self.inputs[i]}"
                time_axis = np.arange(0, time_length, 1) * self.sampling_period
                ax.plot(time_axis, input_array[:, i], drawstyle='steps-pre', label=label)

        for i in range(num_observed_states):
            if i >= len(self.outputs):
                label = f"Observed state: x{i}"
            else:
                label = f"Observed state: {self.outputs[i]}"
            time_axis = np.arange(0, time_length, 1) * self.sampling_period
            ax.plot(time_axis, true_state[:, i], marker='o', label=label, markersize=10)

        for i in range(self.base_model.num_states):
            # Do not show states that are not outputs
            if not show_hidden_states and i >= num_observed_states:
                break
            if i >= len(self.outputs):
                label = f"State: x{i}"
            else:
                label = f"State: {self.outputs[i]}"
            time_axis = np.arange(0, time_length, 1) * self.sampling_period
            ax.plot(time_axis, state_array[0, :, i], marker='.', label=label, markersize=10)
        ax.grid(which="both")
        ax.minorticks_on()

        return state_array

    def get_data_from_dict(self, data_type='input', data_use='training'):
        """
        This function converts data from dictionary and creates a list of torch tensors for easier training.
        :param data_type:
        :param data_use:
        :return:
        """
        data_out = []
        data_source = None
        if data_use == "training":
            data_source = self.training_data
        if data_use == "testing":
            data_source = self.testing_data
        if data_type == 'input':
            data_names = self.inputs
        if data_type == 'output':
            data_names = self.outputs
        for i in range(len(data_source)):
            single_data_out = None
            for data_name in data_names:
                new_element = torch.tensor(data_source[i][data_name]).reshape((-1, 1))
                if single_data_out is None:
                    single_data_out = new_element
                else:
                    single_data_out = torch.cat((single_data_out, new_element), dim=0)
            data_out.append(single_data_out)
        return data_out

    # def learn_base_model(self):
    #     inputs = self.get_input_data("training")
    #     outputs = self.get_output_data("training")
    #     self.learn_grad(self.base_model, inputs, outputs)

    # def learn_error_model(self):
    #     inputs = self.get_input_data("training")
    #     outputs = self.get_output_data("training")
    #     temp_data_ = []
    #     with torch.no_grad():
    #         for i in range(len(inputs)):
    #             out, _ = self.simulate_subsystem(self.base_model, inputs[i].reshape((-1, 1)))
    #             temp_data_.append(out)
    #
    #     self.learn_grad(self.error_model, temp_data_, outputs)

    # def apply_delay(self):
    #     self.start_time = self.delay()
    # def learn_no_grad(self, subsystem):
    #     pass

    def init_learning(self, batch_size):
        self.base_model.init_learning()

    def learn_delay(self, inputs: torch.tensor, true_outputs: torch.tensor, initial_state: torch.tensor = None,
                    batch_size: int = None, optimizer: torch.optim = torch.optim.SGD, learning_rate: int = 0.01,
                    epochs: int = 100, stop_threshold: float = 0.01):
        raise NotImplementedError("learn_delay is not implemented")

    def set_system_history(self,
                           actions: torch.tensor,
                           true_states: torch.tensor or list[torch.tensor],
                           set_base_history: bool = False,
                           set_error_history: bool = False,
                           ):
        """
        :param actions: torch.tensor (BATCH x DATA-1 x NUM_ACTIONS)
        :param true_states: (BATCH x DATA x NUM_OBSERVED_STATES)
        :param set_delay_history: bool
        :param set_base_history: bool
        :param set_error_history: bool
        :return:
        """
        # reset all history
        self.base_model.reset_history()
        self.error_model.reset_history()

        # if set_delay_history:
        #     REQ_DELAY_A_HISTORY = int(self.delay_model.get_delay_val())
        # else:
        #     REQ_DELAY_A_HISTORY = 0

        if set_base_history:
            REQ_BASE_A_HISTORY = self.base_model.reg.action_history_size
            REQ_BASE_S_HISTORY = self.base_model.reg.state_history_size
        else:
            REQ_BASE_A_HISTORY = 1
            REQ_BASE_S_HISTORY = 1

        if set_error_history:
            REQ_ERROR_A_HISTORY = self.error_model.reg.action_history_size
            REQ_ERROR_S_HISTORY = self.error_model.reg.state_history_size
        else:
            REQ_ERROR_A_HISTORY = 1
            REQ_ERROR_S_HISTORY = 1

        # data needs to be synchronized at all time -> we need to take the same history size
        FULL_REQ_HISTORY = max(REQ_BASE_S_HISTORY, REQ_ERROR_S_HISTORY, REQ_BASE_A_HISTORY, REQ_ERROR_A_HISTORY)

        # set start and end of the history correctly for all signals
        A_HISTORY_ERROR_END = S_HISTORY_ERROR_END = A_HISTORY_BASE_END = S_HISTORY_BASE_END = FULL_REQ_HISTORY
        S_HISTORY_BASE_START = FULL_REQ_HISTORY - REQ_BASE_S_HISTORY
        A_HISTORY_BASE_START = FULL_REQ_HISTORY - REQ_BASE_A_HISTORY
        S_HISTORY_ERROR_START = FULL_REQ_HISTORY - REQ_ERROR_S_HISTORY
        A_HISTORY_ERROR_START = FULL_REQ_HISTORY - REQ_ERROR_A_HISTORY
        # A_HISTORY_DELAY_START = FULL_REQ_HISTORY
        # A_HISTORY_DELAY_END = A_HISTORY_DELAY_START + REQ_DELAY_A_HISTORY

        NUM_S_TO_ADD = self.base_model.num_states - true_states.shape[2]
        # TODO fix so the first point is not wasted
        if set_base_history:
            base_hist_s = true_states[:, S_HISTORY_BASE_START:S_HISTORY_BASE_END, :]
            self.base_model.set_history(
                action_history=actions[:, A_HISTORY_BASE_START:A_HISTORY_BASE_END, :],
                state_history=torch.cat(
                    (base_hist_s, torch.zeros((base_hist_s.shape[0], base_hist_s.shape[1], NUM_S_TO_ADD))), dim=-1)
            )

        if set_error_history:
            error_hist_s = true_states[:, S_HISTORY_ERROR_START:S_HISTORY_ERROR_END, :]
            self.error_model.set_history(
                action_history=actions[:, A_HISTORY_ERROR_START:A_HISTORY_ERROR_END, :],
                state_history=torch.cat(
                    (error_hist_s, torch.zeros((error_hist_s.shape[0], error_hist_s.shape[1], NUM_S_TO_ADD))), dim=-1)
            )

        # if set_delay_history:
        #     self.delay_model.set_history(
        #         history=actions[:, A_HISTORY_DELAY_START:A_HISTORY_DELAY_END, :]
        #     )

        actions_no_history = torch.cat((actions[:, FULL_REQ_HISTORY:, :],
                                        torch.zeros((actions.shape[0], actions.shape[1] - FULL_REQ_HISTORY,
                                                     actions.shape[2]))), dim=1)

        true_states_no_history = true_states[:, FULL_REQ_HISTORY:, :]

        return actions_no_history, true_states_no_history

    def nongrad_learn_base_model(self,
                                 inputs: torch.tensor or list[torch.tensor],
                                 true_outputs: torch.tensor or list[torch.tensor],
                                 batch_size: int = None,
                                 optimizer=gfo.EvolutionStrategyOptimizer,
                                 epochs: int = 100,
                                 verbose=False):
        if batch_size is None:
            batch_size = 1
        else:
            raise NotImplementedError('Feel free to finish this')
        self.base_model.batch_size = batch_size
        num_actions = self.base_model.num_actions
        num_states = self.base_model.num_states
        search_space, nongrad_params_flat = self.base_model.get_search_space()

        def objective_function(para):
            # init cumulative loss
            loss = 0.0
            # set model parameters
            for param_name in list(nongrad_params_flat.keys()):
                nongrad_params_flat[param_name].set(para[param_name])

            for i in range(NUM_SIGNALS):
                # reset model memory
                self.base_model.reset()
                # set initial state
                # initial_state = true_outputs[i, 0, 0:self.base_model.num_states]

                NUM_S_TO_ADD = self.base_model.num_states - true_outputs[i].shape[-1]
                initial_state = torch.cat((true_outputs[i][0, :].reshape((1, true_outputs[i].shape[-1])), torch.zeros((
                    batch_size, NUM_S_TO_ADD))), dim=-1)

                state_array, _, _ = self.simulate(input_array=inputs[i],
                                                  initial_state=initial_state,
                                                  use_base_model=True,
                                                  use_error_model=False)

                loss += self.base_model.loss_fn(state_array[0, :, 0:true_outputs[i].shape[-1]], true_outputs[i][1:, :])

            score = -loss
            return score

        if isinstance(inputs, list):
            NUM_SIGNALS = len(inputs)
        elif isinstance(inputs, torch.Tensor):
            if inputs.dim() == 3:
                NUM_SIGNALS = inputs.shape[0]
            elif inputs.dim() == 2:
                NUM_SIGNALS = 1
                inputs = inputs.reshape((NUM_SIGNALS, inputs.shape[0], inputs.shape[1]))
                # TODO fix this
                true_outputs = true_outputs.reshape((NUM_SIGNALS, true_outputs.shape[0], true_outputs.shape[1]))
            else:
                raise ValueError('dimension mismatch')
        else:
            raise TypeError(f'require type list or torch.Tensor but is of type {type(inputs)}')

        opt = optimizer(search_space, population=10000)
        if verbose:
            verbose = ["progress_bar", "print_results", "print_times"]
        opt.search(objective_function, n_iter=epochs, verbosity=verbose)

        for name in list(nongrad_params_flat.keys()):
            nongrad_params_flat[name].set(opt.best_para[name])

    def learn_error_grad(self,
                         inputs: torch.tensor or list[torch.tensor],
                         true_outputs: torch.tensor or list[torch.tensor],
                         batch_size: int = None,
                         optimizer: torch.optim = torch.optim.SGD,
                         learning_rate: int = 0.01,
                         epochs: int = 100,
                         stop_threshold: float = 0.01,
                         verbose=False):
        """
        This function trains the error model.
        :param inputs: list[torch.tensor] or torch.tensor
                    - if torch.tensor:
                        - (BATCH x DATA x NUM_INPUTS)
                    - if list[torch.tensor]
                        - len(list) = BATCH, torch.tensor.size = (DATA x NUM_INPUTS)
                        - where NUM_INPUTS has to be same across whole batch but DATA does not
        :param true_outputs: list[torch.tensor] or torch.tensor
                    - if torch.tensor:
                        - (BATCH x DATA x NUM_OBSERVED_OUTPUTS)
                    - if list[torch.tensor]
                        - len(list) = BATCH, torch.tensor.size = (DATA x NUM_OBSERVED_OUTPUTS)
                        - where NUM_OBSERVED_OUTPUTS has to be same across whole batch but DATA does not
                    Note: DATA must one larger than DATA in inputs for every signal
        :param batch_size: - number of signals in minibatch
        :param optimizer: -
        :param learning_rate: -
        :param epochs: number of learning epochs
        :param stop_threshold: -
        :param verbose
        :return:
        """

        # TODO 00: setup dimensions properly
        if isinstance(inputs, list):
            NUM_SIGNALS = len(inputs)
        elif isinstance(inputs, torch.Tensor):
            if inputs.dim() == 3:
                NUM_SIGNALS = inputs.shape[0]
            elif inputs.dim() == 2:
                NUM_SIGNALS = 1
                inputs = inputs.reshape((NUM_SIGNALS, inputs.shape[0], inputs.shape[1]))
                # TODO fix this
                true_outputs = true_outputs.reshape((NUM_SIGNALS, true_outputs.shape[0], true_outputs.shape[1]))
            else:
                raise ValueError('dimension mismatch')
        else:
            raise TypeError(f'require type list or torch.Tensor but is of type {type(inputs)}')

        train_s = []
        train_u = []
        train_y = []
        with torch.no_grad():
            for i in range(NUM_SIGNALS):
                # 0. Setup history for input delay model and base model
                # actions_no_history, true_states_no_history = self.set_system_history(inputs,
                #                                                                      true_outputs,
                #                                                                      set_base_history=False,
                #                                                                      set_error_history=True,
                #                                                                      )
                actions_no_history = inputs
                true_states_no_history = true_outputs

                # 1. Simulate delay model and base model (get state_base_out)
                NUM_S_TO_ADD = self.base_model.num_states - true_outputs.shape[2]

                init_state = torch.cat((true_states_no_history[:, 0, :], torch.zeros((
                    true_states_no_history.shape[0], NUM_S_TO_ADD))), dim=-1)
                predicted_states, action_delayed, error_array = self.simulate(input_array=actions_no_history,
                                                                              initial_state=init_state,
                                                                              use_base_model=True,
                                                                              use_error_model=False)

                error = true_states_no_history[:, 1:, :] - predicted_states[:, :, :true_outputs.shape[2]]

                train_s.append(predicted_states[0][:-1, :])
                train_u.append(action_delayed[0][1:, :])
                train_y.append(error[0][1:, :])

                # 2. Generate base model error (error = true_output - state_base_out)

        # 3. Set training data for error model  in: state_base_out, inputs, error  out: train_x, train_y
        self.error_model.set_training_data(train_s, train_u, train_y)  # TODO merge these two functions
        train_x_set, train_y_set = self.error_model.init_learning()  # TODO merge these two functions

        # 4. Setup training
        loss_fn = self.error_model.loss_fn
        self.error_model.train()
        optimizer = optimizer(self.error_model.parameters(), lr=learning_rate)

        # 5. Train model
        for i in range(epochs):
            optimizer.zero_grad()
            output, _, _, _, _ = self.error_model(regressors=train_x_set)  # fix this

            loss = -loss_fn(output, train_y_set)

            loss.backward()
            if verbose: print('Iter %d/%d - Loss: %.3f' % (i + 1, epochs, loss.item()))
            optimizer.step()

        # 6. Clean after training
        self.error_model.eval()

        # self.delay_model.reset()
        self.base_model.reset_history()
        self.error_model.reset_history()

    def learn_base_grad(self, inputs: torch.tensor, true_outputs: torch.tensor, initial_state: torch.tensor = None,
                        batch_size: int = None, optimizer: torch.optim = torch.optim.SGD, learning_rate: int = 0.01,
                        epochs: int = 100, stop_threshold: float = 0.01):

        # Check if base model is defined
        if self.base_model is None:
            raise ValueError("Base model is not defined")

        # Check if data is provided, raise error if not
        if true_outputs is None:
            raise ValueError("No true outputs provided")
        if inputs is None:
            raise ValueError("No inputs provided")
        if true_outputs is None and initial_state is None:
            raise ValueError("An initial state must be provided if no true outputs are.")

        # Check that all lists are of the same length
        if len(inputs) != len(true_outputs):
            raise ValueError("Input and output lists must be of the same length")
        if initial_state is not None:
            if len(inputs) != len(initial_state):
                raise ValueError("Input and initial state lists must be of the same length")

        # Within each bag of data, the inputs and outputs must be of the same length or one shorter
        is_equal_length = False
        is_one_shorter = False
        for i in range(len(inputs)):
            # Check if using equal length or one shorter and consistent in all bags
            if inputs[i].shape[0] == true_outputs[i].shape[0]:
                if is_one_shorter:
                    raise ValueError(
                        "Inconsistent data length. Expected inputs to be one shorter than outputs, but found equal length at bag " + str(
                            i))
                is_equal_length = True
            elif inputs[i].shape[0] == true_outputs[i].shape[0] - 1:
                if is_equal_length:
                    raise ValueError(
                        "Inconsistent data length. Expected inputs to be of equal length to outputs, but found one shorter at bag " + str(
                            i))
                is_one_shorter = True
            else:
                raise ValueError("Inputs and outputs must be of the same length or one shorter. Data bag " + str(
                    i) + " failed this requirement")

            # Check if initial state is provided, if so, check that it is only one state
            if initial_state is not None:
                if initial_state[i].shape[0] != self.base_model.num_states:
                    raise ValueError("Initial state must be of the same size as the number of states")

        # Create list of initial states for each bag of data of size 1 x num_states
        if initial_state is None:
            initial_state = []
            if self.base_model.num_states > true_outputs[0].shape[1]:
                # Pad initial state with zeros if it is smaller than the number of states
                for true_output in true_outputs:
                    init_state = torch.zeros((1, self.base_model.num_states))
                    init_state[:, :true_outputs[0].shape[1]] = true_output[0]
                    initial_state.append(init_state)
            else:
                # Construct initial state by concatenating all initial states
                initial_state = [true_output[0] for true_output in true_outputs]
            # Then remove the first element from each true output
            true_outputs = [true_output[1:] for true_output in true_outputs]
            # Only fix inputs if they are of equal length to outputs
            if is_equal_length:
                inputs = [input_[1:] for input_ in inputs]

        NUM_SIGNALS = len(inputs)
        batched_inputs = []
        batched_true_outputs = []
        batched_initial_state = []
        if batch_size is not None:
            if batch_size > NUM_SIGNALS:
                raise ValueError("Batch size must be smaller than number of signals")
            # If batch size is provided, create a list of batches
            for i in range(0, NUM_SIGNALS, batch_size):
                # Get shortest length of all bags of data
                shortest_length = min([input_.shape[0] for input_ in inputs[i:i + batch_size]])
                # Create list of batches of inputs and outputs
                inputs_batch = [input_[:shortest_length] for input_ in inputs[i:i + batch_size]]
                true_outputs_batch = [true_output[:shortest_length] for true_output in true_outputs[i:i + batch_size]]
                initial_state_batch = initial_state[i:i + batch_size]
                # Concatenate all batches into a single tensor
                batched_inputs.append(torch.stack(inputs_batch))
                batched_true_outputs.append(torch.stack(true_outputs_batch))
                batched_initial_state.append(torch.stack(initial_state_batch))
        else:
            # If no batch size is provided, use original data
            batched_inputs = inputs
            batched_true_outputs = true_outputs
            batched_initial_state = initial_state
            batch_size = 1

        self.init_learning(batch_size)

        print("Learning started")
        params = []
        self.base_model.train()
        params += list(self.base_model.parameters())
        optim = optimizer(params, lr=learning_rate)
        loss_func = self.base_model.loss_fn

        if self.error_model is not None:
            self.error_model.eval()

        for epoch in range(epochs):
            print("------")
            loss_above_threshold = False
            for i in range(len(batched_inputs)):
                optim.zero_grad()
                state_array, _, _ = self.simulate(batched_inputs[i], batched_initial_state[i], True, False)
                loss = 0.0
                # if we have more states than the true state, then we only want to compare the first N states
                num_lossy_states = batched_true_outputs[i].shape[-1]
                # if we are modelling with fewer states than the true state, then we only want to compare the first num_states
                if num_lossy_states > self.base_model.num_states:
                    num_lossy_states = self.base_model.num_states

                # One-slice t:t+1 allows us to use the same code for batched and unbatched data while preserving correct dimensions
                loss += loss_func(state_array[..., :num_lossy_states], batched_true_outputs[i][..., :num_lossy_states])

                if loss > stop_threshold:
                    loss_above_threshold = True
                # print loss and iteration number
                print("Iteration: " + str(epoch) + " Loss: " + str(loss.item()))

                loss.backward()
                optim.step()
            if not loss_above_threshold:
                break

        print("Learning finished")
        print("------")

    def randomize_samples(self):
        self.loaded_data = np.random.permutation(self.loaded_data)

    def split_data(self, split_training=0.8):
        if (self.loaded_data is None):
            raise Exception("No data to split")

        self.training_data = []
        self.testing_data = []

        data_num = len(self.loaded_data)

        for i in range(data_num):
            if i / data_num < split_training:
                self.training_data.append(self.loaded_data[i])
            else:
                self.testing_data.append(self.loaded_data[i])

    def save_to_json(self, file_name):

        # Check if folder path exists, raise error if it does not
        file_path = os.path.dirname(file_name)
        if file_path == '':
            file_path = '.'
        if not os.path.exists(file_path):
            raise ValueError("Folder path does not exist")

            # Check if file exists, raise error if it does
        if os.path.isfile(file_name):
            raise ValueError("File already exists")

        json_dict = {}

        json_dict["num_states"] = self.base_model.num_states
        json_dict["num_actions"] = self.base_model.num_actions

        json_dict["sampling_period"] = self.sampling_period

        if self.base_model is not None:
            json_dict["base_model"] = self.base_model.get_json_repr()
        if self.error_model is not None:
            json_dict["error_model"] = self.error_model.get_json_repr()

        # Save to file
        with open(file_name, 'w') as outfile:
            json.dump(json_dict, outfile, sort_keys=False, indent=4)

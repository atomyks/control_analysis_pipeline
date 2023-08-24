from control_analysis_pipeline.model.delay_model.delay_model import InputDelayModel
from control_analysis_pipeline.model.base_model.base_model_linear import BaseLinearModel
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Enum for model selection
class ModelType:
    BASE = 0
    ERROR = 1
    DELAY = 2

class System:
    def __init__(self, loaded_data : dict = None, num_states : int = 1, num_actions : int = 1, sampling_period : float = 0.01):
        if loaded_data is not None:
            self.loaded_data = loaded_data["data"]
        else:
            self.loaded_data = None

        self.training_data = None
        self.testing_data = None
        self.num_states = num_states
        self.num_actions = num_actions

        if loaded_data is not None:
            self.sampling_period = loaded_data["header"]["sampling_period"]
        else:
            self.sampling_period = sampling_period
            
        self.output_data = None

        # system delay
        self.delay_model = InputDelayModel(num_actions=self.num_actions)
        self.base_model = BaseLinearModel(num_actions=self.num_actions, num_states=self.num_states)
        self.error_model = None
        self.inputs = None
        self.outputs = None

        self.loss_fn = nn.MSELoss()

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

        print("-----------")
        print(self.outputs)
        print(self.inputs)

    # def learn_delay(self):
    #     # self.delay.learn()
    #     pass

    def system_step(self, u_input: torch.tensor, state_now: torch.tensor,
                    sim_delay=True, sim_base_model=True, sim_error_model=True) -> (torch.tensor, torch.tensor):
        if sim_delay:
            u_input = self.delay_model(u_input)

        if sim_base_model:
            state_next = self.base_model(u_input, state_now)
        else:
            # if we are not simulating the base model, then the base model is just the identity
            state_next = state_now

        if sim_error_model and self.error_model is not None:
            error = self.error_model(u_input, state_now)
        else:
            error = 0.0

        state_next = state_next + error
        return state_next

    def simulate(self, input_array: torch.tensor, initial_state: torch.tensor = None,
                 use_delay=True, use_base_model=True, use_error_model=True):
        """
        :param input_array: torch.tensor, (BATCH x TIME x NUM_INPUTS) or (TIME x NUM_INPUTS)
        :param initial_state: torch.tensor, (BATCH x NUM_INPUTS)
        :param true_state: torch.tensor, (BATCH x TIME x NUM_TRUE_STATES)
        :param use_delay:
        :param use_base_model:
        :param use_error_model:
        :return:
        """

        if input_array.dim() == 3:
            batch_size, time_length, _ = input_array.shape
        elif input_array.dim() == 2:
            time_length, _ = input_array.shape
            batch_size = 1
        else:
            raise ValueError("Input array must be of shape (BATCH x TIME x NUM_INPUTS) or (TIME x NUM_INPUTS)")


        if initial_state is not None:
            state = initial_state
        else:
            state = torch.zeros((batch_size, self.num_states))

        action_delayed_array = torch.zeros((batch_size, time_length, self.num_actions))
        state_array = torch.zeros((batch_size, time_length, self.num_states))  # BATCH x TIME x STATES
        error_array = torch.zeros((batch_size, time_length, self.num_states))

        for t in range(time_length):
            # One-slice t:t+1 allows us to use the same code for batched and unbatched data while preserving correct dimensions
            state = self.system_step(input_array[..., t, :], state, use_delay, use_base_model, use_error_model)
            state_array[..., t, :] = state    

        return state_array

    def plot_simulation(self, input_array: torch.tensor, initial_state: torch.tensor = None, true_state: torch.tensor = None,
                        ax : plt.Axes = None, show_hidden_states=True, show_input=True,
                        use_delay=True, use_base_model=True, use_error_model=True):
        """
        :param input_array: torch.tensor, (TIME x NUM_INPUTS)
        :param initial_state: torch.tensor, (1 x NUM_STATES)
        :param true_state: torch.tensor, (TIME x NUM_STATES)
        :param use_delay:
        :param use_base_model:
        :param use_error_model:
        :return:
        """
        
        state_array = None
        with torch.no_grad():
            # state_array (TIME x STATES)
            state_array = self.simulate(input_array, 
                                        initial_state=initial_state,
                                        use_delay=use_delay,
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


        for i in range(self.num_states):
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
        self.delay_model.init_learning(batch_size)
        self.base_model.init_learning()

    def learn_delay(self, inputs: torch.tensor, true_outputs: torch.tensor, initial_state: torch.tensor = None, 
                            batch_size: int = None, optimizer: torch.optim = torch.optim.SGD, learning_rate: int = 0.01,
                            epochs: int = 100, stop_threshold: float = 0.01):  
        raise NotImplementedError("learn_delay is not implemented")
    
    def learn_error_grad(self, inputs: torch.tensor, true_outputs: torch.tensor, initial_state: torch.tensor = None, 
                            batch_size: int = None, optimizer: torch.optim = torch.optim.SGD, learning_rate: int = 0.01,
                            epochs: int = 100, stop_threshold: float = 0.01):  
        raise NotImplementedError("learn_base_grad is not implemented")
    
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
                    raise ValueError("Inconsistent data length. Expected inputs to be one shorter than outputs, but found equal length at bag " + str(i))
                is_equal_length = True
            elif inputs[i].shape[0] == true_outputs[i].shape[0] - 1:
                if is_equal_length:
                    raise ValueError("Inconsistent data length. Expected inputs to be of equal length to outputs, but found one shorter at bag " + str(i))
                is_one_shorter = True
            else:
                raise ValueError("Inputs and outputs must be of the same length or one shorter. Data bag " + str(i) + " failed this requirement")
            
            # Check if initial state is provided, if so, check that it is only one state
            if initial_state is not None:
                if initial_state[i].shape[0] != self.num_states:
                    raise ValueError("Initial state must be of the same size as the number of states")
                    
        # Create list of initial states for each bag of data of size 1 x num_states
        if initial_state is None:
            initial_state = []
            if self.num_states > true_outputs[0].shape[1]:
                # Pad initial state with zeros if it is smaller than the number of states
                for true_output in true_outputs:
                    init_state = torch.zeros((1, self.num_states))
                    init_state[:, :true_outputs[0].shape[1]] = true_output[0]
                    initial_state.append(init_state)
            else:
                # Construct initial state by concatenating all initial states
                initial_state =  [true_output[0] for true_output in true_outputs]
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
                shortest_length = min([input_.shape[0] for input_ in inputs[i:i+batch_size]])
                # Create list of batches of inputs and outputs
                inputs_batch = [input_[:shortest_length] for input_ in inputs[i:i+batch_size]]
                true_outputs_batch = [true_output[:shortest_length] for true_output in true_outputs[i:i+batch_size]]
                initial_state_batch = initial_state[i:i+batch_size]
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
                state_array = self.simulate(batched_inputs[i], batched_initial_state[i], True, True, True)
                loss = 0.0
                # if we have more states than the true state, then we only want to compare the first N states
                num_lossy_states = batched_true_outputs[i].shape[-1]
                # if we are modelling with fewer states than the true state, then we only want to compare the first num_states
                if num_lossy_states > self.num_states:
                    num_lossy_states = self.num_states

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
        if(self.loaded_data is None):
            raise Exception("No data to split")
        
        self.training_data = []
        self.testing_data = []

        data_num = len(self.loaded_data)

        for i in range(data_num):
            if i / data_num < split_training:
                self.training_data.append(self.loaded_data[i])
            else:
                self.testing_data.append(self.loaded_data[i])

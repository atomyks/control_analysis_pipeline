from system.delay_model import DelayModel
from system.error_model_demo import ErrorModelDemo
from system.base_model_linear import BaseLinearModel
from system.base_model_feedforward import BaseFeedforwardModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class SystemLearning:
    def __init__(self, loaded_data):
        self.loaded_data = loaded_data["data"]
        self.training_data = None
        self.testing_data = None
        self.split_training = 0.8
        self.batch_size = 1
        self.num_states = 3
        self.num_inputs = 1

        self.sampling_period = loaded_data["header"]["sampling_period"]
        self.output_data = None

        # system delay
        self.delay = DelayModel(batch_size=1, num_inputs=self.num_inputs)
        self.base_model = BaseLinearModel(num_inputs=self.num_inputs, num_outputs=self.num_states)
        self.fd_model = BaseFeedforwardModel(num_inputs=self.num_inputs, num_outputs=self.num_states)
        self.error_model = ErrorModelDemo(num_inputs=self.num_inputs, num_outputs=self.num_states)
        self.inputs = None
        self.outputs = None

        self.loss_fn = nn.MSELoss()

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
            u_input = self.delay(u_input)

        if sim_base_model:
            state_next = self.base_model(u_input, state_now)
        else:
            state_next = self.fd_model(u_input, state_now)

        if sim_error_model:
            error = self.error_model(u_input, state_now)
        else:
            error = 0.0

        state_next = state_next + error
        return state_next

    def simulate(self, input_array: torch.tensor, initial_state: torch.tensor = None, true_state: torch.tensor = None,
                 use_delay=True, use_base_model=True, use_error_model=True):
        """
        :param input_array: torch.tensor, (BATCH x TIME x NUM_INPUTS) or (TIME x NUM_INPUTS)
        :param initial_state: torch.tensor, (BATCH x NUM_INPUTS)
        :param true_state: torch.tensor, (same dims as input_array)
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
            # raise
            return None

        if initial_state is not None:
            state = initial_state
        else:
            state = torch.zeros((batch_size, self.num_states), dtype=torch.float64)
        loss = 0
        state_array = torch.zeros((batch_size, time_length, self.num_states))  # BATCH x TIME x STATES
        for t in range(time_length):
            state = self.system_step(input_array[..., t, :], state, use_delay, use_base_model, use_error_model)
            state_array[..., t, :] = state
            if true_state is not None:
                loss += self.loss_fn(state, true_state[t])
        return state_array, loss

    def plot_simulation(self, input_array: torch.tensor, initial_state: torch.tensor = None,
                        true_state: torch.tensor = None,
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
            state_array, _ = self.simulate(input_array, 
                                        initial_state=initial_state,
                                        use_delay=use_delay,
                                        use_base_model=use_base_model,
                                        use_error_model=use_error_model)

        state_array = state_array.numpy()
        
        if true_state is None:
            num_observed_states = 0
        else:
            _, num_observed_states = true_state.shape
        time_length, num_inputs = input_array.shape

        fig, ax = plt.subplots()
        for i in range(num_inputs):
            label = f"Input: {self.inputs[i]}"
            time_axis = np.arange(0.0, time_length * self.sampling_period, self.sampling_period)
            ax.plot(time_axis, input_array[:, i], '.', label=label)

        for i in range(num_observed_states):
            if i >= len(self.outputs):
                label = f"Observed state: x{i}"
            else:
                label = f"Observed state: {self.outputs[i]}"
            time_axis = np.arange(0.0, time_length * self.sampling_period, self.sampling_period)
            ax.plot(time_axis, true_state[:, i], '.', label=label)

        for i in range(self.num_states):
            if i >= len(self.outputs):
                label = f"State: x{i}"
            else:
                label = f"State: {self.outputs[i]}"
            time_axis = np.arange(0.0, time_length * self.sampling_period, self.sampling_period)
            ax.plot(time_axis, state_array[0, :, i], '.', label=label)
        ax.grid(which="both")
        ax.minorticks_on()
        plt.legend()
        plt.show()

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

    def learn_grad(self, inputs: torch.tensor, initial_state: torch.tensor = None, true_outputs: torch.tensor = None, 
                   optimizer=None, epochs=100, use_delay=True, use_base_model=True, use_error_model=True):
        NUM_SIGNALS = len(inputs)

        params = []
        if use_base_model:
            self.base_model.train()
            params += list(self.base_model.parameters())
        else:
            self.base_model.eval()

        if use_error_model:
            self.error_model.train()
            params += list(self.error_model.parameters())
        else:
            self.error_model.eval()
        
        if len(params) == 0:
            print("No parameters to optimize")
            return
        
        if optimizer is None:
            optimizer = optim.Adam(params, lr=0.01)
        

        # Create list of zero initial states for each bag of data of size 1 x num_states
        if initial_state is None:
            initial_state = torch.zeros((NUM_SIGNALS, self.num_states), dtype=torch.float64)
            
        print("Learning started")
    
        for _ in range(epochs):
            print("------")
            for i in range(NUM_SIGNALS):
                optimizer.zero_grad()
                _, loss = self.simulate(inputs[i], initial_state[i], true_outputs[i], use_delay, use_base_model, use_error_model)
                print(f"Loss: {loss}")
                loss.backward()
                optimizer.step()
        
        print("Learning finished")
        print("------")
        
    def randomize_samples(self):
        self.loaded_data = np.random.permutation(self.loaded_data)

    def split_data(self):
        self.training_data = []
        self.testing_data = []

        data_num = len(self.loaded_data)

        for i in range(data_num):
            if i / data_num < self.split_training:
                self.training_data.append(self.loaded_data[i])
            else:
                self.testing_data.append(self.loaded_data[i])
import torch
import torch.nn as nn
from control_analysis_pipeline.regressor.regressor_factory import RegressorFactory
from control_analysis_pipeline.utils.normalizer import TorchNormalizer
import gpytorch
import gc


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_inputs, num_outputs):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_outputs]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            # gpytorch.kernels.PeriodicKernel(batch_shape=torch.Size([num_outputs]), ard_num_dims=num_inputs) +
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_outputs]), ard_num_dims=num_inputs),
            batch_shape=torch.Size([num_outputs])

        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class ErrorGPModel(nn.Module):
    """
    Similar to https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html
    """

    def __init__(self, num_inputs=1, num_outputs=1):
        super(ErrorGPModel, self).__init__()

        self.back_prop = True
        self.nongrad_params = None

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.init_lengthscale = 0.01

        self.action_history_size = 1
        self.state_history_size = 1
        self.reg = RegressorFactory(batch_size=1, num_actions=self.num_inputs,
                                    num_states=self.num_outputs, action_history_size=self.action_history_size,
                                    state_history_size=self.state_history_size)

        self.gp_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.num_outputs)
        self.gp_model = BatchIndependentMultitaskGPModel(None, None, self.gp_likelihood,
                                                         num_inputs=self.regressor_size(), num_outputs=self.num_outputs)

        self.gp_input = None
        self.gp_output = None
        self.scaler_x = TorchNormalizer(num_of_normalizers=self.num_inputs)
        self.scaler_y = TorchNormalizer(num_of_normalizers=self.num_outputs)

        # Add basic regressors

        s_def = [(0, 0)]
        new_regressor = lambda a, s: s[0]
        self.reg.add(new_regressor, s_defs=s_def)

        # s_def = [(0, 0)]
        # new_regressor = lambda a, s: torch.sin(s[0])
        # self.reg.add(new_regressor, s_defs=s_def)

        # s_def = [(0, 0)]
        # new_regressor = lambda a, s: torch.cos(s[0])
        # self.reg.add(new_regressor, s_defs=s_def)

        a_def = [(0, 0)]  # ------------------------------------------------------------
        new_regressor = lambda a, s: a[0]
        self.reg.add(new_regressor, a_defs=a_def)

        a_def = [(0, 0)]
        new_regressor = lambda a, s: torch.sin(a[0])
        self.reg.add(new_regressor, a_defs=a_def)
        #
        # a_def = [(0, 0)]
        # new_regressor = lambda a, s: torch.cos(a[0])
        # self.reg.add(new_regressor, a_defs=a_def)

        # for i in range(self.action_history_size):
        #     for j in range(self.num_inputs):
        #         a_def = [(-i, j)]
        #         new_regressor = lambda a, s: a[0]
        #         self.reg.add(new_regressor, a_defs=a_def)
        # for i in range(self.state_history_size):
        #     for j in range(self.num_outputs):
        #         s_def = [(-i, j)]
        #         new_regressor = lambda a, s: s[0]
        #         self.reg.add(new_regressor, s_defs=s_def)

    def forward(self,
                regressors: torch.tensor or None = None,
                u_input: torch.tensor or None = None,
                y_last: torch.tensor or None = None):
        """
        :param regressors: torch.tensor, BATCH x NUM_REGRESSORS, GP input
        :param u_input: torch.tensor, BATCH x NUM_INPUTS, system action
        :param y_last: torch.tensor, BATCH x NUM_STATES, system state
        :return:
                 output - gpytorch.distributions.MultitaskMultivariateNormal
                        - .mean (DATA_LENGTH x NUM_OUTPUTS)
                        - .stddev (DATA_LENGTH x NUM_OUTPUTS)
                        - .covariance_matrix (NUM_OUTPUTS * DATA_LENGTH x NUM_OUTPUTS * DATA_LENGTH)
        """
        if regressors is None and (u_input is None or y_last is None):
            raise ValueError("either 'regressors' should be None or both 'u_input' and 'y_last' should be None")
        elif regressors is not None and (u_input is not None or y_last is not None):
            raise ValueError("either 'regressors' should be None or both 'u_input' and 'y_last' should be None")

        if regressors is not None:
            # for training just do predict the model because the input data should be already scaled by
            output, mean, lower, upper, cov = self.predict_model_step(regressors)
            return output, mean, lower, upper, cov
        else:
            regressors = self.get_regressors(u_input, y_last)
            output, mean, lower, upper, cov = self.scale_and_predict_model_step(regressors)
            return output, mean, lower, upper

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
        return len(self.reg)

    def set_training_data(self,
                          train_s: torch.tensor or list,
                          train_a: torch.tensor or list,
                          train_out: torch.tensor or list):  # train_x, train_u, train_y
        """

        Note: BATCH dimension can be either list or torch.tensor. DATA and last  dimension needs to be torch.tensor.
        :param train_s: (BATCH x DATA_LENGTH x NUM_STATES)
        :param train_a: (BATCH x DATA_LENGTH x NUM_ACTIONS)
        :param train_out: (BATCH x DATA_LENGTH x NUM_STATES)
        :return:
        """
        if type(train_s) == list:
            batch_dim = len(train_s)
            if not (len(train_s) == len(train_a) == len(train_out)):
                raise ValueError('dimension mismatch')
        else:
            if train_s.dim() == 2:
                batch_dim = 1
                train_s = train_s.reshape((batch_dim, train_s.shape[0], train_s.shape[1]))
                train_a = train_a.reshape((batch_dim, train_a.shape[0], train_a.shape[1]))
                train_out = train_out.reshape((batch_dim, train_out.shape[0], train_out.shape[1]))
            if train_s.dim() == 3:
                batch_dim, _, _ = train_s.shape
            else:
                raise ValueError('dimension mismatch')

        # data needs to be synchronized at all time -> we need to take the same history size from both signals
        required_history = max(self.action_history_size, self.state_history_size)

        self.gp_input = None
        self.gp_output = None
        idx = 0

        # needs to be done iteratively to support list on the input
        for i in range(batch_dim):
            # get dimensions of the current data signal
            train_s_single = train_s[i]
            data_length_s, num_states = train_s_single.shape
            train_a_single = train_a[i]
            data_length_a, num_actions = train_a_single.shape
            train_out_single = train_out[i]
            data_length_out, num_outputs = train_out_single.shape

            # check dimensions of the current data signal
            if not (data_length_s == data_length_a == data_length_out):
                raise ValueError('dimension mismatch')
            if (not num_states == self.num_outputs) or (not num_actions == self.num_inputs) or (
                    not num_outputs == self.num_outputs):
                raise ValueError('dimension mismatch')

            data_length = data_length_s  # since all of them should be the same it does not matter which one we take

            if self.gp_input is None:
                self.gp_input = torch.zeros((data_length - required_history, self.regressor_size()))
            else:
                self.gp_input = torch.cat(
                    (self.gp_input, torch.zeros((data_length - required_history, self.regressor_size()))
                     ), dim=0)
            if self.gp_output is None:
                self.gp_output = torch.zeros((data_length - required_history, self.num_outputs))
            else:
                self.gp_output = torch.cat(
                    (self.gp_output, torch.zeros((data_length - required_history, self.num_outputs))
                     ), dim=0)

            # set start and end of the history correctly for all signals
            s_history_start = required_history - self.state_history_size
            a_history_start = required_history - self.action_history_size

            # set regressor history
            history_s = train_s_single[s_history_start:required_history, :].reshape((1,
                                                                                     self.state_history_size,
                                                                                     self.num_outputs))
            history_a = train_a_single[a_history_start:required_history, :].reshape((1,
                                                                                     self.action_history_size,
                                                                                     self.num_inputs))
            self.reg.set_history(history_a, history_s)

            # process training data
            for j in range(required_history, data_length):
                regressor = self.get_regressors(u_input=train_a_single[j, :], y_last=train_s_single[j, :])
                self.gp_input[idx, :] = regressor
                self.gp_output[idx, :] = train_out_single[j, :]
                idx += 1

    def scale_and_predict_model_step(self, gp_input):

        point = self.scaler_x.transform(gp_input)

        output, normalized_mean, normalized_lower, normalized_upper, cov = self.predict_model_step(point)

        if len(normalized_lower.shape) == 1:
            normalized_lower = normalized_lower.reshape(1, -1)
            normalized_upper = normalized_upper.reshape(1, -1)

        mean = self.scaler_y.inverse_transform(normalized_mean)
        lower = self.scaler_y.inverse_transform(normalized_lower)
        upper = self.scaler_y.inverse_transform(normalized_upper)

        return output, mean, lower, upper, cov

    def predict_model_step(self, gp_input_normalized):
        """
        :param gp_input_normalized:
        :return: model error prediction
        """
        output = self.gp_likelihood(self.gp_model(gp_input_normalized))
        confidence = output.confidence_region()
        return output, output.mean, torch.squeeze(confidence[0]), torch.squeeze(
            confidence[1]), torch.squeeze(output.stddev) ** 2  # mean, lower, upper

    def init_learning(self):
        train_x = self.gp_input  # (DATA_LENGTH x NUM_REGRESSORS)
        train_y = self.gp_output  # (DATA_LENGTH x NUM_OUTPUTS)

        train_x_scaled = self.scaler_x.fit_transform(train_x)

        train_y_scaled = self.scaler_y.fit_transform(train_y)
        train_y_scaled = train_y_scaled.contiguous()

        if self.gp_likelihood is not None:
            # self.gp_likelihood.cpu()
            del self.gp_likelihood
            gc.collect()
            #  torch.cuda.empty_cache()

        if self.gp_model is not None:
            #  self.gp_model.cpu()
            del self.gp_model
            gc.collect()
            #  torch.cuda.empty_cache()

        self.gp_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.num_outputs)
        self.gp_model = BatchIndependentMultitaskGPModel(train_x_scaled, train_y_scaled, self.gp_likelihood,
                                                         num_inputs=self.regressor_size(), num_outputs=self.num_outputs)

        self.gp_model.covar_module.base_kernel.lengthscale = self.init_lengthscale

        return train_x_scaled, train_y_scaled

    def __str__(self):
        str_out = str(self.reg)
        str_out += "\n"
        str_out += super(ErrorGPModel, self).__str__()
        return str_out


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    torch.set_printoptions(linewidth=100)

    train_x = torch.stack([torch.linspace(0, 1, 100).reshape((100, 1)),
                           torch.linspace(0, 1, 100).reshape((100, 1))
                           ], -1).reshape((100, 2))
    train_u = torch.cos(torch.linspace(0, 1, 100).reshape((100, 1)) * 30.0)

    train_y = torch.stack([
        torch.sin(train_x[:, 0].reshape((100, 1)) * (2 * torch.pi)) + torch.randn(
            train_x[:, 0].reshape((100, 1)).size()) * 0.2 + train_u,
        torch.cos(train_x[:, 0].reshape((100, 1)) * (2 * torch.pi)) + torch.randn(
            train_x[:, 0].reshape((100, 1)).size()) * 0.2 + train_u,
    ], -1).reshape((100, 2))

    # set model
    gp = ErrorGPModel(num_inputs=1, num_outputs=2)

    print(gp)

    # Show training datas
    f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))
    y1_ax.plot(train_x.detach().numpy(), train_y[:, 0].detach().numpy(), 'k*')
    y2_ax.plot(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), 'k*')
    plt.show()

    # TRAIN GP model ------------------------------------------------------------------------------

    # init training
    gp.set_training_data(train_x, train_u, train_y)
    train_x_scaled, train_y_scaled = gp.init_learning()
    training_iterations = 200
    gp.train()
    optimizer = torch.optim.Adam(gp.gp_model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    loss_fun = gpytorch.mlls.ExactMarginalLogLikelihood(gp.gp_likelihood, gp.gp_model)

    # do training iterations
    with gpytorch.settings.cholesky_jitter(1e-1):
        for i in range(training_iterations):
            optimizer.zero_grad()
            output, _, _, _, _ = gp(regressors=train_x_scaled)
            loss = -loss_fun(output, train_y_scaled)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()

    # EVALUATE SYSTEM ------------------------------------------------------------------------------

    # Set into eval mode
    gp.eval()

    simulation_length = 51
    # Initialize plots
    f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))
    predictions = torch.zeros_like(train_y)
    mean = torch.zeros((simulation_length, 2))
    lower = torch.zeros((simulation_length, 2))
    upper = torch.zeros((simulation_length, 2))

    # Make predictions
    with (torch.no_grad(), gpytorch.settings.fast_pred_var()):
        start_sim = -0.5
        end_sim = 1.5
        test_x = torch.stack([torch.linspace(start_sim, end_sim, simulation_length).reshape((simulation_length, 1)),
                              torch.linspace(start_sim, end_sim, simulation_length).reshape((simulation_length, 1))
                              ], -1).reshape((simulation_length, 2))

        test_u = torch.cos(torch.linspace(start_sim, end_sim, simulation_length).reshape((simulation_length, 1)) * 30.0)

        for i in range(51):
            _, mean_, lower_, upper_ = gp(u_input=test_u[i], y_last=test_x[i])  # u_input: torch.tensor, y_last
            mean[i] = mean_
            lower[i] = lower_
            upper[i] = upper_

    # PLOT RESULTS

    # Plot training data as black stars
    y1_ax.plot(train_x.detach().numpy(), train_y[:, 0].detach().numpy(), 'k*')
    # Predictive mean as blue line
    y1_ax.plot(test_x[:, 0].reshape((simulation_length,)).numpy(), mean[:, 0].numpy(), 'b')
    # Shade in confidence
    y1_ax.fill_between(test_x[:, 0].reshape((simulation_length,)).numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(),
                       alpha=0.5)
    y1_ax.set_ylim([-3, 3])
    y1_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y1_ax.set_title('Observed Values (Likelihood)')

    # Plot training data as black stars
    y2_ax.plot(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), 'k*')
    # Predictive mean as blue line
    y2_ax.plot(test_x[:, 0].reshape((simulation_length,)).numpy(), mean[:, 1].numpy(), 'b')
    # Shade in confidence
    y2_ax.fill_between(test_x[:, 0].reshape((simulation_length,)).numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(),
                       alpha=0.5)
    y2_ax.set_ylim([-3, 3])
    y2_ax.legend(['Observed Data', 'Mean', 'Confidence'])
    y2_ax.set_title('Observed Values (Likelihood)')

    plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
from control_analysis_pipeline.models.regressor_factory import RegressorFactory
from control_analysis_pipeline.helpers.normalizer import TorchNormalizer
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
        self.batch_size = 1

        self.action_history_size = 1
        self.state_history_size = 1
        self.reg = RegressorFactory(batch_size=self.batch_size, num_actions=self.num_inputs,
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

        a_def = [(0, 0)]
        new_regressor = lambda a, s: a[0]
        self.reg.add(new_regressor, a_defs=a_def)

        # a_def = [(0, 0)]
        # new_regressor = lambda a, s: torch.sin(a[0])
        # self.reg.add(new_regressor, a_defs=a_def)
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

        # self.lin1 = nn.Linear(num_inputs + num_outputs, 10, dtype=torch.double)
        # self.lin2 = nn.Linear(10, num_outputs, dtype=torch.double)

    def forward(self, u_input: torch.tensor, y_last: torch.tensor or None = None):
        # def forward(self, u_input: torch.tensor):
        """
        :param u_input: torch.tensor, BATCH x INPUTS, system input
        :param y_last: torch.tensor, BATCH x STATES, system input
        :return:
        """
        if self.gp_model.training:
            output, mean, lower, upper, cov = self.predict_model_step(u_input)
            return output
        else:
            regressors = self.get_regressors(u_input, y_last)
            output, mean, lower, upper, cov = self.scale_and_predict_model_step(regressors)
            return output, mean, lower, upper

    def get_regressors(self, u_input: torch.tensor, y_last: torch.tensor):
        """
        :param u_input: torch.tensor, BATCH x INPUTS, system input
        :param y_last: torch.tensor, BATCH x STATES, system input
        :return: computed regressors (BATCH x NUM_REGRESSORS)
        """
        regressors = self.reg(u_input,  # torch.rand((batch_size, num_actions))
                              y_last)  # torch.rand((batch_size, num_states))
        return regressors

    # def set_batch_size(self, new_batch_size):
    #     if new_batch_size > 0:
    #         self.batch_size = 1  # new_batch_size
    #     else:
    #         raise ValueError('Batch size needs to be at least one')

    @staticmethod
    def check_input_data_dims(input_data, last_dim_size):
        """
        :param input_data: train_s: (BATCH x TIME_LENGTH x NUM_XX)
        :param last_dim_size:
        :return:
        """
        if input_data.dim() == 1:
            batch_dim = 1
            data_length = 1
            num_actions = input_data.shape
        elif input_data.dim() == 2:
            batch_dim = 1
            data_length, num_actions = input_data.shape
        elif input_data.dim() == 3:
            batch_dim, data_length, num_actions = input_data.shape
        else:
            raise ValueError('dimension mismatch')
        if num_actions == last_dim_size:
            input_data = input_data.reshape((batch_dim, data_length, num_actions))
        else:
            raise ValueError('dimension mismatch')
        return input_data, batch_dim, data_length

    def regressor_size(self):
        return len(self.reg)

    def set_training_data(self,
                          train_s: torch.tensor,
                          train_a: torch.tensor,
                          train_out: torch.tensor):  # train_x, train_u, train_y
        """
        :param train_s: (BATCH x TIME_LENGTH x NUM_STATES)
        :param train_a: (BATCH x TIME_LENGTH x NUM_ACTIONS)
        :param train_out: (BATCH x TIME_LENGTH x NUM_STATES)
        :return:
        """
        train_a, batch_dim, data_length = self.check_input_data_dims(train_a, self.num_inputs)
        train_s, _, _ = self.check_input_data_dims(train_s, self.num_outputs)
        train_out, _, _ = self.check_input_data_dims(train_out, self.num_outputs)

        required_history = max(self.action_history_size, self.state_history_size)

        self.gp_input = torch.zeros(
            ((data_length - required_history) * batch_dim, self.regressor_size()))  # (DATA_LENGTH x NUM_INPUTS)
        self.gp_output = torch.zeros(
            ((data_length - required_history) * batch_dim, self.num_outputs))  # (DATA_LENGTH x NUM_OUTPUTS)
        idx = 0

        for i in range(batch_dim):
            s_history_start = required_history - self.state_history_size
            a_history_start = required_history - self.action_history_size

            history_s = train_s[i][s_history_start:required_history, :].reshape((1,
                                                                                 self.state_history_size,
                                                                                 self.num_outputs))
            history_a = train_a[i][a_history_start:required_history, :].reshape((1,
                                                                                 self.action_history_size,
                                                                                 self.num_inputs))

            self.reg.set_history(history_a, history_s)
            for j in range(required_history, data_length):
                regressor = self.get_regressors(u_input=train_a[i][j, :], y_last=train_s[i][j, :])
                self.gp_input[idx, :] = regressor
                self.gp_output[idx, :] = train_out[i][j, :]
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

    def init_gp(self):
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

    gp = ErrorGPModel(num_inputs=1, num_outputs=2)

    print(gp)

    # Show training datas
    f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))
    y1_ax.plot(train_x.detach().numpy(), train_y[:, 0].detach().numpy(), 'k*')
    y2_ax.plot(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), 'k*')
    plt.show()

    # TRAIN GP model

    gp.set_training_data(train_x, train_u, train_y)
    train_x_scaled, train_y_scaled = gp.init_gp()
    training_iterations = 100
    # Find optimal model hyperparameters
    gp.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(gp.gp_model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    # "loss_fun" for GPs - the marginal log likelihood
    loss_fun = gpytorch.mlls.ExactMarginalLogLikelihood(gp.gp_likelihood, gp.gp_model)
    with gpytorch.settings.cholesky_jitter(1e-1):
        for i in range(training_iterations):
            optimizer.zero_grad()
            output = gp(train_x_scaled)
            # output, mean, lower, upper, cov = gp.scale_and_predict_model_step(train_x_scaled)

            # train_x_scaled - np.array (DATA_LENGTH x NUM_OUTPUTS)

            # output - gpytorch.distributions.MultitaskMultivariateNormal
            #        - .mean (DATA_LENGTH x NUM_OUTPUTS)
            #        - .stddev (DATA_LENGTH x NUM_OUTPUTS)
            #        - .covariance_matrix (NUM_OUTPUTS * DATA_LENGTH x NUM_OUTPUTS * DATA_LENGTH)
            # print(f"Scaled output covariance_matrix: {output.covariance_matrix}")

            loss = -loss_fun(output, train_y_scaled)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()

    # EVALUATE SYSTEM

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
        # test_x = torch.linspace(0, 1, 51)
        start_sim = -1.0
        end_sim = 2.0
        test_x = torch.stack([torch.linspace(start_sim, end_sim, simulation_length).reshape((simulation_length, 1)),
                              torch.linspace(start_sim, end_sim, simulation_length).reshape((simulation_length, 1))
                              ], -1).reshape((simulation_length, 2))

        test_u = torch.cos(torch.linspace(start_sim, end_sim, simulation_length).reshape((simulation_length, 1)) * 30.0)

        for i in range(51):
            _, mean_, lower_, upper_ = gp(test_u[i], test_x[i])  # u_input: torch.tensor, y_last
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

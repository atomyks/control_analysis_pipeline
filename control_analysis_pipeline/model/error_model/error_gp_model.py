import torch
import torch.nn as nn
from control_analysis_pipeline.model.error_model.error_model import ErrorModel
from control_analysis_pipeline.utils.normalizer import TorchNormalizer
import gpytorch
import gc
from typing import Optional

class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_actions, num_states):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_states]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            # gpytorch.kernels.PeriodicKernel(batch_shape=torch.Size([num_states]), ard_num_dims=num_actions) +
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_states]), ard_num_dims=num_actions),
            batch_shape=torch.Size([num_states])

        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class ErrorGPModel(ErrorModel):
    """
    Similar to https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html
    """

    def __init__(self, num_actions=1, num_states=1, action_history_size=1, state_history_size=1):
        super(ErrorGPModel, self).__init__(num_actions=num_actions, num_states=num_states, action_history_size=action_history_size, state_history_size=state_history_size)

        self.gp_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.num_states)
        self.gp_model = BatchIndependentMultitaskGPModel(None, None, self.gp_likelihood,
                                                         num_actions=self.regressor_size(), num_states=self.num_states)
        self.init_lengthscale = 0.1
        
        self.scaler_x = TorchNormalizer(num_of_normalizers=self.num_actions)
        self.scaler_y = TorchNormalizer(num_of_normalizers=self.num_states)

        # Add basic regressors
        s_def = [(0, 0)]
        new_regressor = lambda a, s: s[0]
        self.reg.add(new_regressor, s_defs=s_def)

        a_def = [(0, 0)]
        new_regressor = lambda a, s: a[0]
        self.reg.add(new_regressor, a_defs=a_def)

    def forward(self,
                regressors: Optional[torch.tensor] = None,
                u_input: Optional[torch.tensor] = None,
                y_last: Optional[torch.tensor] = None):
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
        train_x = self.model_input  # (DATA_LENGTH x NUM_REGRESSORS)
        train_y = self.model_output  # (DATA_LENGTH x NUM_OUTPUTS)

        train_x_scaled = self.scaler_x.fit_transform(train_x)

        train_y_scaled = self.scaler_y.fit_transform(train_y)
        train_y_scaled = train_y_scaled.contiguous()

        if self.gp_likelihood is not None:
            del self.gp_likelihood
            gc.collect()

        if self.gp_model is not None:
            del self.gp_model
            gc.collect()

        self.gp_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.num_states)
        self.gp_model = BatchIndependentMultitaskGPModel(train_x_scaled, train_y_scaled, self.gp_likelihood,
                                                         num_actions=self.regressor_size(), num_states=self.num_states)

        self.gp_model.covar_module.base_kernel.lengthscale = self.init_lengthscale

        # Initialize cost function for the GP
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_likelihood, self.gp_model)
        self.enable_grad_learning(mll)

        return train_x_scaled, train_y_scaled

    def get_json_repr(self):
        '''
        
        :return: json representation of the model
        '''
        json_repr = super(ErrorGPModel, self).get_json_repr()
        
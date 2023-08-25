import torch
import gpytorch
from control_analysis_pipeline.model.error_model.error_gp_model import ErrorGPModel

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
    gp = ErrorGPModel(num_actions=1, num_states=2, num_errors=2)

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

    # Save model to json
    json_repr = gp.get_json_repr()
    import json
    print(json.dumps(json_repr, sort_keys=False, indent=4))
    
    # Save to file
    with open('gp_example.json', 'w') as outfile:
        json.dump(json_repr, outfile, sort_keys=False, indent=4)
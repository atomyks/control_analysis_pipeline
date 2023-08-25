import torch
import torch.nn as nn
from control_analysis_pipeline.model.model import Model
from control_analysis_pipeline.functional.deadzone import Deadzone
import os
from control_analysis_pipeline.model.base_model.base_model_simple_steering import SimpleSteering
import matplotlib.pyplot as plt
from control_analysis_pipeline.system.system import System
# from gradient_free_optimizers import EvolutionStrategyOptimizer
import gradient_free_optimizers as gfo


def deadzone_autoware(input, br, bl, brr, bll):
    res = 0.0
    if input >= br:
        res = input - br * brr
    if bl < input < br:
        res = 0.0
    if input <= bl:
        res = input - bl * bll
    return res


def system_get_f(x, J, T_in, b3, T_f):
    x1, x2 = x

    x1_dot = x2
    x2_dot = 1 / J * (T_in - b3 * x2 - T_f)

    x_dot = torch.tensor([x1_dot, x2_dot])
    return x_dot


def nongrad_learn_base_model(inputs: torch.tensor or list[torch.tensor],
                             true_outputs: torch.tensor or list[torch.tensor],
                             batch_size: int = None,
                             optimizer=gfo.EvolutionStrategyOptimizer,
                             epochs: int = 100,
                             verbose=False):
    if batch_size is None:
        batch_size = 1
    system.base_model.batch_size = batch_size
    num_actions = system.base_model.num_actions
    num_states = system.base_model.num_states

    def objective_function(para):
        # init cumulative loss
        loss = 0.0
        # set model parameters
        for param_name in list(nongrad_params_flat.keys()):
            nongrad_params_flat[param_name].set(para[param_name])
        # reset model memory
        system.base_model.reset()
        # set initial state
        initial_state = torch.zeros((batch_size, num_actions))

        for i in range(NUM_SIGNALS):
            state_array, _, _ = system.simulate(input_array=inputs[i], initial_state=initial_state,
                                                use_delay=True, use_base_model=True,
                                                use_error_model=False)

            loss += loss_fn(state_array[i], true_outputs[i, 1:, 0:state_array.shape[-1]])

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

    opt = optimizer(search_space)
    if verbose:
        verbose = ["progress_bar", "print_results", "print_times"]
    opt.search(objective_function, n_iter=epochs, verbosity=verbose)

    # exit()

    for name in list(nongrad_params_flat.keys()):
        nongrad_params_flat[name].set(opt.best_para[name])


if __name__ == "__main__":
    # -----------------------------------GENERATE TRAINING DATA---------------------------------------------
    # define model parameters
    Bl = 0.0081  # 0.00178745
    Br = 0.0088  # 0.00178745
    D = 0
    e = 0.87
    e2 = 0.8

    Kp = 121.0
    Ki = 0.008
    Kd = 0.038
    J = 0.7
    b3 = 8.2
    T_f = 0.0
    brr = 1.0
    bll = 1.5

    # simulation parameters
    end_time = 100.0
    dt = 0.1

    time_axis = torch.linspace(0, end_time, int(end_time / dt))
    time_axis = time_axis.reshape((time_axis.shape[0], 1))
    cmd = true_u = torch.cat((
        torch.sin(torch.ones((time_axis[0:100].shape[0], 1)) * time_axis[0:100]) * 0.01,
        torch.sign(torch.sin(torch.ones((time_axis[100:200].shape[0], 1)) * time_axis[100:200])) * 0.004,
        torch.sin(torch.ones((time_axis[200:].shape[0] - 1, 1)) * time_axis[200:][:-1]) * torch.cumsum(torch.randn(
            (time_axis[200:].shape[0] - 1, 1)) * 0.0002, dim=0),
    ), dim=0)
    true_s = torch.zeros((time_axis.shape[0], 2))

    state = torch.tensor([10., 10.])
    true_s[0, :] = state

    u_virt = torch.zeros((cmd.shape[0] - D,))
    last_steer_rate = 0.0
    y_ax = torch.zeros_like(cmd)

    x1, x2 = 0.0, 0.0
    u_virt[0] = x1
    x = torch.tensor([x1, x2])
    integral = 0.0
    last_error = 0.0

    for i in range(cmd.shape[0] - 1 - D):
        a = 0.0
        if i > 1000:
            a = 0.0
            cmd[i] += a
        error = -(u_virt[i] - (cmd[i]))
        integral += error
        T_in = error * Kp + integral * Ki + (last_error - error) * Kd
        last_error = error
        x = x + system_get_f(x, J, T_in, b3, T_f) * 0.03  # Forward euler
        x[1] = deadzone_autoware(x[1], Br, -Bl, brr, bll)
        true_s[i, 0] = x[0]
        true_s[i, 1] = x[1]
        u_virt[i + 1] = x[0]
        last_steer_rate = x[1] * e

    show_plots = True

    if show_plots:
        # Show training datas
        f, (y1_ax, y2_ax) = plt.subplots(2, 2, figsize=(8, 3))
        y1_ax[0].plot(time_axis[:-1].detach().numpy(), true_u[:, 0].detach().numpy(), 'g', label="u1")
        y1_ax[0].plot(time_axis.detach().numpy(), true_s[:, 0].detach().numpy(), 'k', label="angle")
        y1_ax[0].set_xlim((0.0, 50.0))

        y1_ax[1].plot(time_axis.detach().numpy(), true_s[:, 1].detach().numpy(), 'b', label="speed")
        y1_ax[1].set_xlim((0.0, 50.0))
        y1_ax[0].legend()
        y1_ax[1].legend()

    loss_fn = nn.L1Loss()
    # -----------------------------------GENERATE TRAINING DATA DONE---------------------------------------------

    system = System(num_states=1, num_actions=2)
    system.base_model = SimpleSteering()
    print("--------------------------TEST--------------------------")
    search_space, nongrad_params_flat = system.base_model.get_search_space()

    nongrad_learn_base_model(inputs=true_u,
                             true_outputs=true_s,
                             batch_size=None,
                             optimizer=gfo.EvolutionStrategyOptimizer,
                             epochs=100,
                             verbose=True)

    steer_predicted = torch.zeros((true_u.shape[0] + 1, true_u.shape[1]))
    steer = torch.tensor([0.0])
    steer_predicted[0] = steer
    system.base_model.reset()
    for i in range(true_u.shape[0]):
        steer = system.base_model(a_input=true_u[i].reshape((1, 1)), y_last=steer.reshape((1, 1)))
        steer_predicted[i + 1] = steer

    if show_plots:
        # Show training datas
        y2_ax[0].plot(time_axis[:-1].detach().numpy(), true_u[:, 0].detach().numpy(), 'g', label="u1")
        y2_ax[0].plot(time_axis.detach().numpy(), steer_predicted[:, 0].detach().numpy(), 'k', label="angle")
        y2_ax[0].set_xlim((0.0, 50.0))
        y2_ax[0].legend()
        plt.show()

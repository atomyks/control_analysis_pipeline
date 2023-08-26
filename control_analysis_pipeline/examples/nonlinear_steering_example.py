import torch
import torch.nn as nn
from control_analysis_pipeline.model.base_model.base_model_simple_steering import SimpleSteering
import matplotlib.pyplot as plt
from control_analysis_pipeline.system.system import System
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

    # -----------------------------------GENERATE TRAINING DATA DONE---------------------------------------------

    # Learn system
    system = System(num_states=1, num_actions=1)
    system.base_model = SimpleSteering()
    print("--------------------------TEST--------------------------")

    system.nongrad_learn_base_model(inputs=true_u,
                                    true_outputs=true_s,
                                    optimizer=gfo.EvolutionStrategyOptimizer,
                                    epochs=200,
                                    verbose=True)

    # Simulate learned system
    initial_state = torch.zeros((1, 1))
    system.base_model.reset()
    steer_predicted, _, _ = system.simulate(input_array=true_u, initial_state=initial_state,
                                            use_delay=True, use_base_model=True,
                                            use_error_model=False)

    if show_plots:
        # Show training datas
        y2_ax[0].plot(time_axis[:-1].detach().numpy(), true_u[:, 0].detach().numpy(), 'g', label="u1")
        y2_ax[0].plot(time_axis[1:].detach().numpy(), steer_predicted[0][:, 0].detach().numpy(), 'k', label="angle")
        y2_ax[0].set_xlim((0.0, 50.0))
        y2_ax[0].legend()
        plt.show()

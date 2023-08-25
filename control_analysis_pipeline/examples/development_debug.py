import torch
import gpytorch
from control_analysis_pipeline.model.error_model.error_gp_model import ErrorGPModel
from control_analysis_pipeline.system.system import System
import matplotlib.pyplot as plt


def testing_model(a, s, dt):
    """
    :param a:
    :param s:
    :return:
    """
    s1, s2 = s

    s_next = torch.zeros((2,))
    s_next[0] = s1 + (torch.sin(s1) + a) * dt
    s_next[1] = s2 + (torch.cos(s1) - s2) * dt

    return s_next


if __name__ == "__main__":

    sys = System(num_actions=1, num_states=3)

    sys.delay_model.set_delay_val(5)



    # sys.base_model.reg.state_history_size = 4
    # sys.base_model.reg.action_history_size = 4
    # for i in range(4):
    #     s_def = [(-i, 0)]
    #     new_regressor = lambda a, s: s[0]
    #     sys.base_model.reg.add(new_regressor, s_defs=s_def)
    # for i in range(4):
    #     a_def = [(-i, 0)]
    #     new_regressor = lambda a, s: a[0]
    #     sys.base_model.reg.add(new_regressor, a_defs=a_def)

    print("-----------------Printing base model-----------------")
    print(sys.base_model)

    sys.error_model = ErrorGPModel(num_actions=1, num_states=3, num_errors=2,
                                   action_history_size=8,
                                   state_history_size=8)

    for i in range(8):
        s_def = [(-i, 0)]
        new_regressor = lambda a, s: s[0]
        sys.error_model.reg.add(new_regressor, s_defs=s_def)
    for i in range(8):
        a_def = [(-i, 0)]
        new_regressor = lambda a, s: a[0]
        sys.error_model.reg.add(new_regressor, a_defs=a_def)
    print("-----------------Printing error model-----------------")
    print(sys.error_model)

    end_time = 100.0
    dt = 0.1

    time_axis = torch.linspace(0, end_time, int(end_time / dt))
    time_axis = time_axis.reshape((time_axis.shape[0], 1))
    true_u = torch.zeros((time_axis.shape[0] - 1, 1))
    true_s = torch.zeros((time_axis.shape[0], 2))

    state = torch.tensor([10., 10.])
    true_s[0, :] = state

    for i in range(true_u.shape[0]):
        action = torch.sin(time_axis[i, 0]) * 2.0
        true_u[i, :] = action

        state = testing_model(a=action, s=state, dt=dt)
        true_s[i + 1, :] = state
    # true_u = torch.sign(torch.cos(time_axis[:-1] * 15.0))
    #
    # time_now = 0.0
    #
    # for i in range(5000):
    #     dt = 0.1
    #     testing_model(a, s)
    #
    #
    #
    # true_s = torch.stack([torch.sin(time_axis * 15.0),
    #                       torch.sin(time_axis * 15.0) * 1.2], -1).reshape((100, 2))

    show_plots = True

    if show_plots:
        # Show training datas
        f, (y1_ax, y2_ax) = plt.subplots(2, 2, figsize=(8, 3))
        y1_ax[0].plot(time_axis[:-1].detach().numpy(), true_u[:, 0].detach().numpy(), 'g', label="u1")
        y1_ax[0].plot(time_axis.detach().numpy(), true_s[:, 0].detach().numpy(), 'k', label="s1")
        y1_ax[0].plot(time_axis.detach().numpy(), true_s[:, 1].detach().numpy(), 'b', label="s2")
        y1_ax[0].legend()

        y2_ax[1].plot(true_s[:, 0].detach().numpy(), true_s[:, 1].detach().numpy(), 'b', label="s1 x s2")

        # plt.show()

    # exit()

    # PIPELINE TESTING

    # learn base model

    import copy

    sys.learn_base_grad(inputs=[true_u], true_outputs=[copy.deepcopy(true_s)], initial_state=None,
                        batch_size=None,
                        optimizer=torch.optim.Adam,
                        learning_rate=0.00002,
                        epochs=120,
                        stop_threshold=0.001)

    # Learn error model
    sys.learn_error_grad(inputs=true_u, true_outputs=true_s,
                         verbose=True, epochs=1000,
                         optimizer=torch.optim.Adam,
                         learning_rate=0.008)

    true_outputs = copy.deepcopy(true_s).reshape((1, true_s.shape[0], true_s.shape[1]))

    # simulate base model
    actions_no_history, true_states_no_history = sys.set_system_history(
        copy.deepcopy(true_u).reshape((1, true_u.shape[0], true_u.shape[1])),
        true_outputs,
        set_delay_history=False,
        set_base_history=False,
        set_error_history=True,
    )

    NUM_S_TO_ADD = sys.num_states - true_outputs.shape[2]

    init_state = torch.cat((true_states_no_history[:, 0, :], torch.zeros((
        true_states_no_history.shape[0], NUM_S_TO_ADD))), dim=-1)

    predicted_states, action_delayed, error_array = sys.simulate(input_array=actions_no_history,
                                                                 initial_state=init_state,
                                                                 use_delay=True,
                                                                 use_base_model=True,
                                                                 use_error_model=False)

    print(predicted_states.shape)

    if show_plots:
        y1_ax[1].plot(time_axis[:-1].detach().numpy(), true_u[:, 0].detach().numpy(), 'g', label="u1")
        y1_ax[1].plot(time_axis.detach().numpy()[2:], predicted_states[0, :, 0].detach().numpy(), 'k', label="u2")
        y1_ax[1].plot(time_axis.detach().numpy()[2:], predicted_states[0, :, 1].detach().numpy(), 'b', label="s1")
        y1_ax[1].legend()
        # plt.show()

    # simulate base + error model

    true_outputs = copy.deepcopy(true_s).reshape((1, true_s.shape[0], true_s.shape[1]))
    actions_no_history, true_states_no_history = sys.set_system_history(
        copy.deepcopy(true_u).reshape((1, true_u.shape[0], true_u.shape[1])),
        true_outputs,
        set_delay_history=False,
        set_base_history=False,
        set_error_history=False,
    )

    NUM_S_TO_ADD = sys.num_states - true_outputs.shape[2]

    init_state = torch.cat((true_states_no_history[:, 0, :], torch.zeros((
        true_states_no_history.shape[0], NUM_S_TO_ADD))), dim=-1)

    predicted_states, action_delayed, error_array = sys.simulate(input_array=copy.deepcopy(true_u).reshape((1, true_u.shape[0], true_u.shape[1])),
                                                                 initial_state=init_state,
                                                                 use_delay=True,
                                                                 use_base_model=True,
                                                                 use_error_model=True)

    print(predicted_states.shape)

    if show_plots:
        y2_ax[0].plot(time_axis[:-1].detach().numpy(), true_u[:, 0].detach().numpy(), 'g', label="u1")
        y2_ax[0].plot(time_axis.detach().numpy()[0:],
                      torch.cat((true_states_no_history[:, 0, 0], predicted_states[0, :, 0])).detach().numpy(), 'k',
                      label="s1")

        y2_ax[0].plot(time_axis.detach().numpy()[1:],
                      error_array[0, :, 1].detach().numpy(), 'p',
                      label="e1")

        y2_ax[0].plot(time_axis.detach().numpy()[0:],
                      torch.cat((true_states_no_history[:, 0, 1], predicted_states[0, :, 1])).detach().numpy(), 'b',
                      label="s2")
        y2_ax[0].legend()
        plt.show()

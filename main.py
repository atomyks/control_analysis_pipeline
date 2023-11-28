import argparse

import gpytorch.settings
from control_analysis_pipeline.data_preprocessing.data_preprocessor import DataPreprocessor
from control_analysis_pipeline.system.system import System
import torch
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Parse arguments for launching from command line
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", help="Path to a config file"
    )
    args = parser.parse_args()
    data_preprocessor = DataPreprocessor()

    # Load config file
    data_preprocessor.parse_config(config_name=args.config)

    # Load data
    data = data_preprocessor.load()

    # Define system
    sys = System(data)
    sys.parse_config(config_name=args.config)

    sys.randomize_samples()
    sys.split_data(split_training=1.0)

    input_train_arr = sys.get_data_from_dict('input', 'training')
    outputs_train_arr = sys.get_data_from_dict('output', 'training')
    input_test_arr = sys.get_data_from_dict('input', 'testing')
    outputs_test_arr = sys.get_data_from_dict('output', 'testing')

    print("pipeline start")

    print(f"Number of training examples: {len(input_train_arr)}")
    print(f"Number of training examples: {len(input_test_arr)}")
    # print(input_train_arr[0][:-1, :].shape)
    # print(input_train_arr[1][:-1, :].shape)
    #
    # print(len(outputs_train_arr))
    # print(outputs_train_arr[0].shape)
    # print(outputs_train_arr[1].shape)

    from control_analysis_pipeline.model.base_model.base_model_steering_v2 import Steeringv2
    from control_analysis_pipeline.model.base_model.base_model_simple_steering import SimpleSteering
    from control_analysis_pipeline.model.base_model.base_model_simple_steering_histeresis import SimpleSteeringHist
    from control_analysis_pipeline.model.error_model.error_gp_model import ErrorGPModel
    import gradient_free_optimizers as gfo
    import copy
    # Learn system
    # sys.base_model = Steeringv2(dt=0.03)
    # sys.base_model = SimpleSteering(dt=0.03)
    sys.base_model = SimpleSteeringHist(dt=0.03)

    # hst = 30
    # 
    # sys.error_model = ErrorGPModel(num_actions=1, num_states=sys.base_model.num_states, num_errors=1,
    #                                action_history_size=hst*2,
    #                                state_history_size=hst*2)
    # 
    # for i in range(hst):
    #     # # Add basic regressors
    #     s_def = [(-i*2, 0)]
    #     new_regressor = lambda a, s: s[0]
    #     sys.error_model.reg.add(new_regressor, s_defs=s_def)

    #     a_def = [(-i*2, 0)]
    #     new_regressor = lambda a, s: a[0]
    #     sys.error_model.reg.add(new_regressor, a_defs=a_def)

    #     s_def = [(-i*2, 0)]
    #     a_def = [(-i*2, 0)]
    #     new_regressor = lambda a, s: a[0] - s[0]
    #     sys.error_model.reg.add(new_regressor, a_defs=a_def, s_defs=s_def)
    #
    # s_def = [(-1, 0)]
    # new_regressor = lambda a, s: s[0]
    # sys.error_model.reg.add(new_regressor, s_defs=s_def)
    #
    # a_def = [(-1, 0)]
    # new_regressor = lambda a, s: a[0]
    # sys.error_model.reg.add(new_regressor, a_defs=a_def)

    train = True
    if train:
        train_a = []
        for i in range(len(input_train_arr)):
            train_a.append(input_train_arr[i][:-1, :])

        sys.nongrad_learn_base_model(
            inputs=copy.deepcopy(train_a),  # [:-1, :]
            true_outputs=copy.deepcopy(outputs_train_arr),
            optimizer=gfo.ParticleSwarmOptimizer,
            epochs=100,
            verbose=True)
    else:
        sys.base_model.deadzone_layer.r_lambd.set(0.0020)
        sys.base_model.deadzone_layer.l_lambd.set(-0.0021)
        sys.base_model.deadzone_layer.sc.set(1.0)
        sys.base_model.time_const.set(0.2)
        sys.base_model.delay_layer.set_delay_val(3)

    # # Learn error model
    # sys.learn_error_grad(inputs=copy.deepcopy(train_a),
    #                      true_outputs=copy.deepcopy(outputs_train_arr),
    #                      verbose=True, epochs=1,
    #                      optimizer=torch.optim.Adam,
    #                      learning_rate=0.005)

    fig, axs = plt.subplots(3, 3)
    # initial_state = torch.cat((outputs_train_arr[0][0:1, :], torch.zeros((1, 1))), dim=-1)
    for i in range(3):
        for j in range(3):
            sys.base_model.reset()
            initial_state = outputs_train_arr[i*3+j][0:1, :]
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                sys.plot_simulation(input_array=copy.deepcopy(input_train_arr[i*3+j][:-1, :]),
                                    true_state=copy.deepcopy(outputs_train_arr[i*3+j][1:, :]),
                                    initial_state=initial_state,
                                    ax=axs[i][j], show_input=True, show_hidden_states=False,
                                    use_base_model=True, use_error_model=False)
    

    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.show()


if __name__ == "__main__":
    main()

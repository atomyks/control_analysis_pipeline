import argparse
from yaml import load
from yaml import Loader
from control_analysis_pipeline.data_preprocessing.data_preprocessor import DataPreprocessor
from control_analysis_pipeline.system.system import System
import torch
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", help="Path to a config file"
    )
    args = parser.parse_args()
    data_preprocessor = DataPreprocessor()

    config = None
    with open(args.config, "r") as stream:
        config = load(stream, Loader=Loader)

    data_preprocessor.parse_config(config)

    print(f"---------------Data loading start--------------")
    data_preprocessor.load_data()
    print(f"---------------Data loading done---------------")
    data_preprocessor.sync_data()
    # data_preprocessor.plot_all_data(["steer_status", "steer_cmd"])
    data_preprocessor.filter_data_on_enable()
    # data_preprocessor.plot_all_data(["steer_status", "steer_cmd"])
    # data_preprocessor.save_all_data(["steer_status", "steer_cmd"])

    data = data_preprocessor.get_preprocessed_data()

    sys = System(data, num_states=2, num_actions=1)
    sys.parse_config(config)

    sys.randomize_samples()
    sys.split_data()

    input_train_arr = sys.get_data_from_dict('input', 'training')
    input_test_arr = sys.get_data_from_dict('input', 'testing')
    outputs_train_arr = sys.get_data_from_dict('output', 'training')

    print("pipeline start")

    print(len(input_train_arr))
    print(len(input_test_arr))
    print(input_train_arr[0][:-1, :].shape)
    print(input_train_arr[1][:-1, :].shape)

    print(len(outputs_train_arr))
    print(outputs_train_arr[0].shape)
    print(outputs_train_arr[1].shape)

    from control_analysis_pipeline.model.base_model.base_model_steering_v2 import Steeringv2
    from control_analysis_pipeline.model.base_model.base_model_simple_steering import SimpleSteering
    import gradient_free_optimizers as gfo
    import copy
    # Learn system
    # system = System(num_states=2, num_actions=1)
    # sys.base_model = Steeringv2(dt=0.03)
    sys.base_model = SimpleSteering(dt=0.03)

    train_a = []
    for i in range(len(input_train_arr)):
        train_a.append(input_train_arr[i][:-1, :])

    sys.nongrad_learn_base_model(
        inputs=copy.deepcopy(train_a),  # [:-1, :]
        true_outputs=copy.deepcopy(outputs_train_arr),
        optimizer=gfo.ParticleSwarmOptimizer,
        epochs=200,
        verbose=True)

    fig, axs = plt.subplots(2, 2)

    # initial_state = torch.cat((outputs_train_arr[0][0:1, :], torch.zeros((1, 1))), dim=-1)
    initial_state = outputs_train_arr[0][0:1, :]
    sys.plot_simulation(input_array=copy.deepcopy(input_train_arr[0][:-1, :]),
                        true_state=copy.deepcopy(outputs_train_arr[0][1:, :]),
                        initial_state=initial_state,
                        ax=axs[0][0], show_input=True, show_hidden_states=False,
                        use_base_model=True, use_error_model=False)

    # initial_state = torch.cat((outputs_train_arr[1][0:1, :], torch.zeros((1, 1))), dim=-1)
    initial_state = outputs_train_arr[1][0:1, :]
    sys.plot_simulation(input_array=copy.deepcopy(input_train_arr[1][:-1, :]),
                        true_state=copy.deepcopy(outputs_train_arr[1][1:, :]),
                        initial_state=initial_state,
                        ax=axs[0][1], show_input=True, show_hidden_states=False,
                        use_base_model=True, use_error_model=False)

    # initial_state = torch.cat((outputs_train_arr[2][0:1, :], torch.zeros((1, 1))), dim=-1)
    initial_state = outputs_train_arr[2][0:1, :]
    sys.plot_simulation(input_array=copy.deepcopy(input_train_arr[2][:-1, :]),
                        true_state=copy.deepcopy(outputs_train_arr[2][1:, :]),
                        initial_state=initial_state,
                        ax=axs[1][0], show_input=True, show_hidden_states=False,
                        use_base_model=True, use_error_model=False)

    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.show()

    exit()

    # Train the model on the bags of data
    sys.learn_grad(inputs=input_train_arr, true_outputs=outputs_train_arr,
                   batch_size=2, optimizer=torch.optim.SGD, learning_rate=0.2,
                   epochs=100, use_delay=False, use_base_model=True, use_error_model=False)

    print('Learned A: ', sys.base_model.A.weight)
    print('Learned B: ', sys.base_model.B.weight)

    # Plot simulation results of first 2 bags of data
    fig, axs = plt.subplots(1, 2)
    for bag_idx, ax in enumerate(axs.ravel()):
        input_arr = torch.from_numpy(sys.loaded_data[bag_idx][sys.inputs[0]]).reshape(
            (sys.loaded_data[bag_idx][sys.inputs[0]].shape[0], 1))
        output_arr = torch.from_numpy(sys.loaded_data[bag_idx][sys.outputs[0]]).reshape(
            (sys.loaded_data[bag_idx][sys.outputs[0]].shape[0], 1))

        sys.plot_simulation(input_array=input_arr,
                            true_state=output_arr,
                            ax=ax, show_input=True, show_hidden_states=False,
                            use_base_model=True, use_error_model=False)
    # top right legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.show()


if __name__ == "__main__":
    main()

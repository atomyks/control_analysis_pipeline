import argparse
from yaml import load
from yaml import Loader
from control_analysis_pipeline.data_preprocessing.data_preprocessor import DataPreprocessor
from control_analysis_pipeline.system.system import SystemLearning
import torch


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

    sys = SystemLearning(data)
    sys.parse_config(config)

    sys.randomize_samples()
    sys.split_data()

    inputs_arr = sys.get_data_from_dict('input', 'training')
    outputs_arr = sys.get_data_from_dict('output', 'training')

    # Train the model on the bags of data
    sys.learn_grad(inputs=inputs_arr, true_outputs=outputs_arr, epochs=100, use_delay=True, use_base_model=True, use_error_model=False)

    bag_idx = 0
    input_arr = torch.from_numpy(sys.loaded_data[bag_idx][sys.inputs[0]]).reshape(
        (sys.loaded_data[bag_idx][sys.inputs[0]].shape[0], 1))
    output_arr = torch.from_numpy(sys.loaded_data[bag_idx][sys.outputs[0]]).reshape(
        (sys.loaded_data[bag_idx][sys.outputs[0]].shape[0], 1))
    
    print(input_arr.shape)
    sys.plot_simulation(input_array=input_arr,
                        true_state=output_arr,
                        use_delay=True, use_base_model=True, use_error_model=False)

    # u_init -> BATCH x history x inputs
    # x_last -> BATCH x history x states

    # u = torch.tensor([[[0.1], [0.1]], [[0.1], [0.1]]])
    # x = torch.tensor([[[0.1], [0.1]], [[0.1], [0.1]]])
    #
    #
    # # u = torch.tensor([[0.1], [0.1]])
    # # x = torch.tensor([[0.1], [0.1]])
    #
    #
    # print("********")
    # print(u.shape)
    # print(u.dim())
    # print(x.shape)
    #
    # f = lambda u_input, x_last: torch.vstack([
    #     u_input[..., 0, 0] + x_last[..., 0, 0],
    #     u_input[..., 0, 0] + x_last[..., 0, 0] * 2.0,
    #     u_input[..., 0, 0] + x_last[..., 0, 0] * 20.0,
    # ])
    #
    # y = f(u_input=u, x_last=x)
    #
    # print(y.shape)
    #
    # print(y)

    # sys.learn_base_model()
    # sys.learn_error_model()

    # from system.base_model_linear import BaseLinearModel
    #
    # base_model = BaseLinearModel()
    #
    # print(base_model)
    # print(base_model.parameters())
    #
    # for param in base_model.parameters():
    #     print(type(param), param.size())
    #
    # print("--------------------------------------------")
    # from system.delay_model_sample import DelayModelSample
    # print("Delay model")
    # delay = DelayModelSample()
    # print(delay)
    #
    # for param in delay.parameters():
    #     print(type(param), param.size())
    #
    # print(delay.nongrad_params.keys())
    # sys.learn_delay()


if __name__ == "__main__":
    main()

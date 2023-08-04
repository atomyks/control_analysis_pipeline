import argparse
from yaml import load, dump
from yaml import Loader, Dumper
from control_analysis_tool import ControlAnalysisTool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", help="Path to a config file"
    )
    args = parser.parse_args()
    control_analysis_tool = ControlAnalysisTool()

    with open(args.config, "r") as stream:
        data = load(stream, Loader=Loader)
        control_analysis_tool.parse_config(data)

    print(f"---------------Data loading start--------------")
    control_analysis_tool.load_data()
    print(f"---------------Data loading done---------------")
    control_analysis_tool.sync_data()
    control_analysis_tool.plot_all_data(["steer_status", "steer_cmd"])
    control_analysis_tool.filter_data_on_enable()
    control_analysis_tool.plot_all_data(["steer_status", "steer_cmd"])
    control_analysis_tool.save_all_data(["steer_status", "steer_cmd"])


if __name__ == "__main__":
    main()

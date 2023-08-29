import numpy as np

import matplotlib.pyplot as plt
from control_analysis_pipeline.data_preprocessing.data_loader import extract_data, get_data_structure
from control_analysis_pipeline.data_preprocessing.signal_synchronization import sync_signal_sample_rates
from pathlib import Path
from control_analysis_pipeline.data_preprocessing.data_filtering import filer_signal_on_enabled


class DataPreprocessor:
    def __init__(self):
        self.in_files_dir = None
        self.out_files_dir = None
        self.resampling_period = None

        self.loaded_data = None
        self.minimum_signal_length = 0.0
        self.discard_after_enable = 0.5
        self.data_to_load = {}

    def parse_config(self, config):
        self.in_files_dir = Path(config["in_files_dir"])
        self.out_files_dir = Path(config["out_files_dir"])
        self.data_to_load = config["data_to_load"]
        self.resampling_period = config["resampling_period"]
        self.minimum_signal_length = config["minimum_signal_length"]
        self.discard_after_enable = config["discard_after_enable"]

    def get_file_names_to_load(self, suffix='.mcap'):
        return sorted(self.in_files_dir.glob(f'**/*{suffix}'))

    def load_data(self, suffix='.mcap'):
        self.loaded_data = []
        file_names = self.get_file_names_to_load(suffix)
        if suffix == '.mcap':
            for file_name in file_names:
                print(f"Extracting data from the bag: {file_name.name}", end='', flush=True)
                extracted_data = extract_data(self.data_to_load, file_name)

                if extracted_data is None:
                    print(f" > ERROR: No data in a topic - skipping the bag")
                    continue
                print(f" > OK")
                self.loaded_data.append(extracted_data)

            if len(self.loaded_data) == 0:
                print(f"ERROR: No valid data found in any of bags")
                return

            # Convert lists -> np.array
            for i in range(len(self.loaded_data)):
                for key_data_name in list(self.loaded_data[i].keys()):
                    for key_atr_name in list(self.loaded_data[i][key_data_name].keys()):
                        self.loaded_data[i][key_data_name][key_atr_name] = np.array(
                            self.loaded_data[i][key_data_name][key_atr_name])
        else:
            # Not implemented error
            pass

    def sync_data(self):
        for i in range(len(self.loaded_data)):
            data = []
            for key in list(self.loaded_data[i].keys()):
                data.append((self.loaded_data[i][key]["time_stamp"],
                             self.loaded_data[i][key]["data"]))

            res = sync_signal_sample_rates(data, sampling_time=self.resampling_period, rm_time_offset=True)

            for j, key in enumerate(list(self.loaded_data[i].keys())):
                self.loaded_data[i][key]["time_stamp"] = res[0]
                self.loaded_data[i][key]["data"] = res[j + 1]

    def filter_data_on_enable(self):
        """
        This function filters data that are gathered while car is in manual mode.
        Warning! This function assumes that the data are synchronized and sampled with period = "self.resampling_period"
        :return:
        """

        filtered_data_arr = []
        for i in range(len(self.loaded_data)):
            data = None
            for key in list(self.loaded_data[i].keys()):
                if data is None:
                    data = self.loaded_data[i][key]["data"]
                else:
                    data = np.vstack((data, self.loaded_data[i][key]["data"]))

            data_enabled = None
            for key in list(self.loaded_data[i].keys()):
                if self.data_to_load[key]["type"] == "enable":
                    if data_enabled is None:
                        data_enabled = self.loaded_data[i][key]["data"]
                    else:
                        data_enabled = np.logical_and(data_enabled, self.loaded_data[i][key]["data"])

            res_arr = filer_signal_on_enabled(data,
                                              data_enabled,
                                              dt=self.resampling_period,
                                              discard_after_enable=self.discard_after_enable,
                                              minimum_signal_length=self.minimum_signal_length)
            for k in range(len(res_arr)):
                temp_dict = {}
                for j, key in enumerate(list(self.loaded_data[i].keys())):
                    temp_dict[key] = get_data_structure()
                    temp_dict[key]["data"] = res_arr[k][j]
                    temp_dict[key]["time_stamp"] = np.arange(0.0, res_arr[k][j].shape[0] * self.resampling_period,
                                                             self.resampling_period)

                filtered_data_arr.append(temp_dict)

        self.loaded_data = filtered_data_arr

    def plot_data(self, idx: int, names: list = None):
        fig, ax = plt.subplots()
        if names is None:
            names = list(self.loaded_data[idx].keys())
        for i, key in enumerate(names):
            ax.plot(self.loaded_data[idx][key]["time_stamp"], self.loaded_data[idx][key]["data"], '.', label=key)
        ax.grid(which="both")
        ax.minorticks_on()
        plt.legend()
        plt.show()

    def get_preprocessed_data(self):
        data_all = {"header":
            {
                "sampling_period": self.resampling_period
            },
            "data": []}
        for i in range(len(self.loaded_data)):
            single_data = {}
            for j, key in enumerate(list(self.loaded_data[i].keys())):
                single_data[key] = self.loaded_data[i][key]["data"]
            data_all["data"].append(single_data)
        return data_all

    def plot_all_data(self, names_to_plot: list = None):
        for i in range(len(self.loaded_data)):
            self.plot_data(i, names_to_plot)

    def save_data(self, idx: int, names: list = None):
        if names is None:
            names = list(self.loaded_data[idx].keys())
        header = ""
        for i, name in enumerate(names):
            if i == len(names) - 1:
                header += f"{name}"
            else:
                header += f"{name}, "

        time_axis = self.loaded_data[idx][names[0]]["time_stamp"]
        data = time_axis
        for i, key in enumerate(names):
            data = np.vstack((data, self.loaded_data[idx][key]["data"]))
        data = data.T
        np.savetxt(f"{self.out_files_dir}/data_{idx}.csv", data, fmt='%.5e',
                   delimiter=",", header=header)

    def save_all_data(self, names_to_plot: list = None):
        for i in range(len(self.loaded_data)):
            self.save_data(i, names_to_plot)

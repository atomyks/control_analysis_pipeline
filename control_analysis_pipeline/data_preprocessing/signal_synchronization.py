import numpy as np


def sync_signal_sample_rates(signals: list, sampling_time: float, rm_time_offset=False) -> list:
    """
    :param signals:
        [(t1, y1), (t2, y2), ..., (tN, yN)]
        tn, yn ... np.array
    :param sampling_time:
    :param rm_time_offset:
    :return:
        size: (N + 1) x NumOfOutSamples
        np.array([t, y1, y2, ..., yN])
    """
    # TODO remove "interpolation" for bool values
    start_time = -np.infty
    end_time = np.infty
    for signal in signals:
        if signal[0][0] > start_time:
            start_time = signal[0][0]
        if signal[0][-1] < end_time:
            end_time = signal[0][-1]

    num_of_out_samples = int((end_time - start_time) / sampling_time + 1)

    res = np.zeros((len(signals) + 1, num_of_out_samples))

    # Create time axis
    res[0] = np.arange(start_time, end_time, sampling_time)

    for signal, signal_id in zip(signals, range(len(signals))):
        original_index = 1
        current_index = 0
        current_time = res[0][current_index]
        signal_time = signal[0]
        signal_data = signal[1]
        while True:
            if signal_time[original_index] > current_time:
                new_y = np.interp(
                    current_time,
                    [signal_time[original_index - 1], signal_time[original_index]],
                    [signal_data[original_index - 1], signal_data[original_index]]
                )
                res[signal_id + 1][current_index] = new_y
                current_index += 1
                if current_index >= num_of_out_samples:
                    break
                current_time = res[0][current_index]
            else:
                original_index += 1

    if rm_time_offset:
        res[0] = np.arange(0.0, end_time - start_time, sampling_time)

    return res

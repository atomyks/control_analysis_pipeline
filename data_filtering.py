import numpy as np


def find_runs(value, a):
    isvalue = np.concatenate(([0], np.not_equal(a, value).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isvalue))

    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def filer_signal_on_enabled(input_data: np.array,
                            enable_signals: np.array,
                            dt: float,
                            discard_after_enable: float = 0.0,
                            minimum_signal_length: float = 0.0) -> list:
    """
    :param input_data: n x m
    :param enable_signals: m
    :param dt:
    :param discard_after_enable:
    :param minimum_signal_length:
    :return:
    """
    out_arr = []
    ranges = find_runs(0, enable_signals)

    for i in range(len(ranges)):
        ranges[i][0] += int(discard_after_enable / dt)
        if ranges[i][1] - ranges[i][0] > minimum_signal_length / dt:
            new_arr = input_data[:, ranges[i][0]:ranges[i][1]]
            out_arr.append(new_arr)
        else:
            continue
    return out_arr

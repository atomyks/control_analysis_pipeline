from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


def read_messages(input_bag, topics: list):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(input_bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    topic_types = reader.get_all_topics_and_types()

    def typename(topic_name):
        for topic_type in topic_types:
            if topic_type.name == topic_name:
                return topic_type.type
        raise ValueError(f"topic {topic_name} not in bag")

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if not topic in topics:
            continue
        try:
            msg_type = get_message(typename(topic))
        except:
            continue
        try:
            msg = deserialize_message(data, msg_type)
        except:
            continue
        yield topic, msg, timestamp
    del reader


def get_data_structure():
    data_structure = {
        "data": [],
        "time_stamp": [],
    }
    return data_structure


def extract_data_mcap(data_to_load, file_name):
    # Extract required topics from the file
    extracted_data = {}
    name_dict = {}

    for signal in list(data_to_load.keys()):

        signal_topic = data_to_load[signal]["topic"]
        signal_var = data_to_load[signal]["var"]

        # check if the signal topic exists
        if signal_topic == "":
            continue

        if not signal_topic in list(name_dict.keys()):
            name_dict[signal_topic] = {}
            name_dict[signal_topic][signal_var] = [signal]
        else:
            name_dict[signal_topic][signal_var].append(signal)

        extracted_data[signal] = get_data_structure()

    topics_to_get = list(name_dict.keys())

    for topic, msg, timestamp in read_messages(file_name, topics=topics_to_get):
        for val_name in list(name_dict[topic].keys()):
            val_name_split = val_name.split('.')
            val = msg
            for attr in val_name_split:
                val = getattr(val, attr)
            for i in range(len(name_dict[topic][val_name])):
                extracted_data[name_dict[topic][val_name][i]]["data"].append(val)
                extracted_data[name_dict[topic][val_name][i]]["time_stamp"].append(timestamp / 1000000000.0)  # Convert to [s]

    for signal_name in list(data_to_load.keys()):
        if len(extracted_data[signal_name]["data"]) == 0:
            return None
    return extracted_data

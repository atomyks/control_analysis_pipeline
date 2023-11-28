import os
import subprocess
import yaml
from tqdm import tqdm
from pqdm.threads import pqdm
import time
from pathlib import Path, PurePath
import zstandard


def decompress_zstandard_to_folder(input_file: Path, destination_dir: Path) -> None:
    input_file = Path(input_file)
    with open(input_file, 'rb') as compressed:
        decomp = zstandard.ZstdDecompressor()
        output_path = Path(destination_dir).joinpath(input_file.stem)
        with open(output_path, 'wb') as destination:
            decomp.copy_stream(compressed, destination)


def preprocess_bags(zip_file_path: Path, delete_original_zip: bool, delete_db: bool) -> None:
    yaml_config_data = {
        "output_bags": [
            {
                "uri": 'mcap_bag',
                "storage_id": 'mcap',
                "all": True,
            },
        ]
    }

    # get all necessary paths
    db_bag_name = zip_file_path.stem
    bag_path = zip_file_path.parent
    yaml_file_path = bag_path.joinpath(".config.yaml")
    db_file_path = bag_path.joinpath(db_bag_name)
    yaml_config_data["output_bags"][0]["uri"] = str(bag_path.joinpath(f"mcap_{Path(db_bag_name).stem}"))

    if zip_file_path.suffix == ".zst":
        decompress_zstandard_to_folder(zip_file_path, destination_dir=zip_file_path.parent)
    else:
        return

    # create .yaml config
    with open(yaml_file_path, 'w') as f:
        yaml.dump(yaml_config_data, f, default_flow_style=False)

    # ros2 bag convert -i <path>/<file>.db3 -o <path>/config.yaml
    # TODO rewrite to use ros2 bag API if exists
    subprocess.run(["ros2", "bag", "convert", "-i", str(db_file_path), "-o", str(yaml_file_path)], capture_output=True)

    # Clean folders after each creation
    if delete_db:
        os.remove(db_file_path)

    os.remove(yaml_file_path)

    if delete_original_zip:
        os.remove(zip_file_path)


def run_preprocess_bags(folder: PurePath, delete_original_zip: bool, delete_db: bool, threading: bool) -> None:
    # Get all .zst bag_files
    rosbags_zips = sorted(Path(folder).glob('**/*.zst'))
    if threading:
        pass
        # args = rosbags_zips
        # result = pqdm(args, process_bags, n_jobs=4)  # 2 * os.cpu_count()
        print("Error not implemented")
        return
    else:
        pbar = tqdm(rosbags_zips)
        for zip_path in pbar:
            preprocess_bags(zip_path, delete_original_zip, delete_db)


if __name__ == "__main__":
    f = "testing"
    f = PurePath(Path.cwd()).joinpath(f)
    del_zip = True
    del_db3 = True
    threads = False

    run_preprocess_bags(f, del_zip, del_db3, threads)

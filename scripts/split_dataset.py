# Examples of how to run the command in CLI:

# Example 1: Basic usage with default output paths
# python scripts/split_dataset.py path/to/preprocessed_features.pt path/to/config1.json path/to/config2.json

# Example 2: Specifying custom output paths
# python scripts/split_dataset.py path/to/preprocessed_features.pt path/to/config1.json path/to/config2.json --output_paths path/to/output1_filtered.pt path/to/output2_filtered.pt

# Example 3: Using multiple config files
# python scripts/split_dataset.py path/to/preprocessed_features.pt path/to/config1.json path/to/config2.json path/to/config3.json

# Example 4: Using wildcard for multiple config files
# python scripts/split_dataset.py path/to/preprocessed_features.pt path/to/config*.json

import argparse
import json
import os
from typing import Dict, List

import torch


def load_config(config_path: str) -> Dict[str, Dict[str, any]]:
    with open(config_path) as f:
        config = json.load(f)
    return config


def filter_preprocessed_features(data: Dict, config_dict: Dict[str, Dict[str, List[str]]]) -> Dict:
    print("Filtering data...")
    filtered_indices = []
    new_label_names = []

    for i, (key, _, _) in enumerate(data["index"]):
        if key in config_dict:
            filtered_indices.append(i)
            new_label_names.append(config_dict[key]["mood/theme"])

    filtered_data = {
        "source": data["source"][filtered_indices],
        "label_names": new_label_names,
        "index": [data["index"][i] for i in filtered_indices],
    }

    print("Filtering complete.")
    return filtered_data


def process_configs(file_path: str, config_paths: List[str], output_paths: List[str]):
    print(f"Loading file: {file_path}")
    data = torch.load(file_path)

    for config_path, output_path in zip(config_paths, output_paths):
        config_dict = load_config(config_path)
        filtered_data = filter_preprocessed_features(data, config_dict)

        print(f"Saving filtered features to {output_path}...")
        torch.save(filtered_data, output_path)
        print(f"Filtered data saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter preprocessed features.")
    parser.add_argument("file_path", type=str, help="Path to the preprocessed features file")
    parser.add_argument("config_paths", nargs="+", type=str, help="Paths to the config JSON files")
    parser.add_argument(
        "--output_paths", nargs="*", type=str, help="Optional output paths for filtered data"
    )
    args = parser.parse_args()

    if not args.output_paths:
        args.output_paths = [
            args.file_path.replace(
                ".pt", f"_{config_path.split('/')[-1].split('.')[0]}_filtered.pt"
            )
            for config_path in args.config_paths
        ]
    else:
        print(args.output_paths)
        if len(args.output_paths) == 1 and os.path.isdir(args.output_paths[0]):
            output_dir = args.output_paths[0]
            args.output_paths = [
                os.path.join(
                    output_dir,
                    os.path.basename(args.file_path).replace(
                        ".pt", f"_{config_path.split('/')[-1].split('.')[0]}_filtered.pt"
                    ),
                )
                for config_path in args.config_paths
            ]
        elif len(args.output_paths) != len(args.config_paths):
            raise ValueError(
                "Number of output paths must match number of config paths when specifying file paths."
            )

    for output_path in args.output_paths:
        if os.path.exists(output_path):
            print(f"Warning: Output file already exists: {output_path}")
            print("This may result in overwriting existing data.")
            user_input = input(
                "Do you want to continue? (y/n/s) [y: yes, n: no, s: skip this file]: "
            ).lower()
            if user_input == "n":
                print("Operation aborted by user.")
                exit(0)
            elif user_input == "s":
                print(f"Skipping {output_path}")
                args.output_paths.remove(output_path)
                args.config_paths.pop(args.output_paths.index(output_path))

    process_configs(args.file_path, args.config_paths, args.output_paths)

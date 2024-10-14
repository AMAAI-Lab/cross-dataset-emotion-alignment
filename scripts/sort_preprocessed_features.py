import argparse
import os

import torch


def sort_preprocessed_features(file_path):
    print(f"Loading file: {file_path}")
    # Load the file
    data = torch.load(file_path)

    source = data["source"]
    label_names = data["label_names"]
    index = data["index"]

    print("Sorting index and reordering...")
    # Sort the index based on the first element of each tuple
    sorted_indices = sorted(range(len(index)), key=lambda i: index[i][0])

    sorted_source = source[sorted_indices]
    sorted_label_names = [label_names[i] for i in sorted_indices]
    sorted_index = [index[i] for i in sorted_indices]

    sorted_data = {
        "source": sorted_source,
        "label_names": sorted_label_names,
        "index": sorted_index,
    }

    print("Sorting complete.")
    return sorted_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort preprocessed features.")
    parser.add_argument("file_path", type=str, help="Path to the preprocessed features file")
    parser.add_argument("--output_path", type=str, help="Optional output path for sorted data")
    args = parser.parse_args()

    # Save the sorted data
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = args.file_path.replace(".pt", "_sorted.pt")

    if os.path.exists(output_path):
        print(f"Output file already exists: {output_path}")
        print("Aborting to prevent overwriting existing data.")
        exit(1)

    sorted_data = sort_preprocessed_features(args.file_path)
    print("Saving sorted features...")
    torch.save(sorted_data, output_path)
    print(f"Sorted data saved to: {output_path}")

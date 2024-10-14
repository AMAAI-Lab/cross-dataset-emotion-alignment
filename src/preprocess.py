import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from src.data.segmented_dataset import JamendoSegmentedDataset
from util import LabelFeatureMapper


def ddp_setup(rank: int, world_size: int) -> None:
    """Set up the Distributed Data Parallel (DDP) environment.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
    """
    os.environ["MASTER_ADDR"] = "0.0.0.0"
    os.environ["MASTER_PORT"] = "10056"
    init_process_group(backend="nccl", world_size=world_size, rank=rank)


def main(rank: int, world_size: int, path_template: str) -> None:
    """Main function to run the preprocessing on a single GPU.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        path_template (str): Template for the output file path.
    """
    ddp_setup(rank, world_size)
    preprocess_mert_features(rank, path_template)
    destroy_process_group()


class PreprocessingJamendoDataset(JamendoSegmentedDataset):
    """Extended JamendoSegmentedDataset for preprocessing purposes.

    Adds index information to each item.
    """

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset with additional index information.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'source': Processed audio segment
                - 'label_names': List of mood/theme labels
                - 'targets': Feature vector representation of labels
                - 'index': Tuple of (key, segment_start_time, segment_end_time)
        """
        key, segment_start_time = self.index[idx]
        item = super().__getitem__(idx)
        item["index"] = (key, segment_start_time, segment_start_time + self.segment_duration)
        return item

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for creating batches with index information.

        Args:
            batch (List[Dict[str, Any]]): List of individual data samples.

        Returns:
            Dict[str, Any]: A dictionary containing batched data with index information.
        """
        source = torch.stack([torch.from_numpy(item["source"]) for item in batch])
        label_names = [item["label_names"] for item in batch]
        index = [item["index"] for item in batch]
        return {"source": source, "label_names": label_names, "index": index}


def preprocess_mert_features(gpu_id: int, path_template: str) -> None:
    """Preprocess audio features using the MERT model.

    Args:
        gpu_id (int): ID of the GPU to use for processing.
        path_template (str): Template for the output file path.
    """
    label_mapper = LabelFeatureMapper.from_npz("resources/label_features_emotify.npz")
    metadata = json.load(open("resources/emotify_metadata_updated.json"))
    audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M")
    dataset = PreprocessingJamendoDataset(
        metadata, "datasets/emotifymusic_wav", audio_processor, 10, label_mapper
    )
    data_loader = DataLoader(
        dataset,
        collate_fn=PreprocessingJamendoDataset.collate_fn,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
        sampler=DistributedSampler(dataset),
    )
    audio_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(gpu_id)

    source: List[torch.Tensor] = []
    label_names: List[List[str]] = []
    index: List[Tuple[str, float, float]] = []

    audio_model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Processing on GPU {gpu_id}"):
            hidden_states = audio_model(
                batch["source"].squeeze(1).to(gpu_id), output_hidden_states=True
            ).hidden_states
            audio_features = torch.stack([h.detach()[:, 0, :] for h in hidden_states], dim=1)
            source.append(audio_features.cpu())
            label_names.extend(batch["label_names"])
            index.extend(batch["index"])

    # Concatenate features
    source_tensor = torch.cat(source, dim=0)

    # Save preprocessed data
    save_dict = {"source": source_tensor, "label_names": label_names, "index": index}
    torch.save(save_dict, path_template.format(gpu_id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MERT features")
    parser.add_argument("audio_duration", type=int, help="Duration of audio segments in seconds")
    args = parser.parse_args()

    output_folder = f"resources/mert_features_emotify_{args.audio_duration}s"
    path_template = f"{output_folder}/partition_{{0}}.pt"
    combined_file_path = path_template.format("combined")

    # # Check if the output folder and combined file already exist
    # if os.path.exists(output_folder) and os.path.exists(combined_file_path):
    #     print(f"Preprocessed features already exist at {combined_file_path}. Exiting.")
    #     exit(0)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, path_template), nprocs=world_size)

    # After all processes finish, combine the preprocessed features
    combined_source: List[torch.Tensor] = []
    combined_label_names: List[List[str]] = []
    combined_index: List[Tuple[str, float, float]] = []

    for i in range(world_size):
        file_path = path_template.format(i)
        if os.path.exists(file_path):
            data = torch.load(file_path)
            combined_source.append(data["source"])
            combined_label_names.extend(data["label_names"])
            combined_index.extend(data["index"])
        # os.remove(file_path)  # Uncomment to remove individual files after combining

    combined_source_tensor = torch.cat(combined_source, dim=0)
    save_dict = {
        "source": combined_source_tensor,
        "label_names": combined_label_names,
        "index": combined_index,
    }
    torch.save(save_dict, combined_file_path)
    print(f"Combined preprocessed features saved to {combined_file_path}")

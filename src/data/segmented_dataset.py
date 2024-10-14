import json
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.data.dataset import AudioMoodDataset
from util import LabelFeatureMapper


class JamendoSegmentedDataset(AudioMoodDataset):
    """A dataset class for segmented audio data from the Jamendo dataset.

    This class extends AudioMoodDataset to handle segmented audio files, allowing for processing of
    fixed-duration segments from longer audio tracks.
    """

    def __init__(
        self,
        data_dict: Dict[str, Dict],
        root_dir: str,
        audio_processor,
        segment_duration: float,
        label_feature_mapper,
    ):
        """Initialize the JamendoSegmentedDataset.

        Args:
            data_dict (Dict[str, Dict]): Dictionary containing metadata for each audio file.
            root_dir (str): Root directory containing the audio files.
            audio_processor: Processor for audio feature extraction.
            segment_duration (float): Duration of each audio segment in seconds.
            label_feature_mapper: Mapper for converting labels to feature vectors.
        """
        self.data = data_dict
        self.root_dir = root_dir
        self.audio_processor = audio_processor
        self.processor_sampling_rate = self.audio_processor.sampling_rate
        self.segment_duration = segment_duration
        self.keys = list(data_dict.keys())
        self.label_feature_mapper = label_feature_mapper
        self.index = self._build_index()

    def __len__(self) -> int:
        """Return the total number of segments in the dataset."""
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """Get a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'source': Processed audio segment
                - 'label_names': List of mood/theme labels
                - 'targets': Feature vector representation of labels
        """
        key, segment_start_time = self.index[idx]
        item = self.data[key]

        audio_path = os.path.join(self.root_dir, item["path"])
        info = torchaudio.info(audio_path)
        src_sampling_rate = info.sample_rate

        # Load the audio segment
        frame_offset = int(segment_start_time * src_sampling_rate)
        num_frames = int(self.segment_duration * src_sampling_rate)
        audio, _ = torchaudio.load(audio_path, num_frames=num_frames, frame_offset=frame_offset)

        # Resample if necessary
        if src_sampling_rate != self.processor_sampling_rate:
            audio = T.Resample(src_sampling_rate, self.processor_sampling_rate)(audio)

        # Pad or cut the audio to match the target length
        target_length = int(self.segment_duration * self.processor_sampling_rate)
        if audio.shape[1] < target_length:
            audio = torch.nn.functional.pad(audio, (0, target_length - audio.shape[1]))
        elif audio.shape[1] > target_length:
            audio = audio[:, :target_length]

        # Convert to mono if necessary
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Process the audio
        processed_audio = self.audio_processor(audio, sampling_rate=self.processor_sampling_rate)[
            "input_values"
        ][0]

        return {
            "source": processed_audio,
            "label_names": item["mood/theme"],
            "targets": self.label_feature_mapper.get_features(item["mood/theme"]),
        }

    def _build_index(self) -> List[Tuple[str, float]]:
        """Build an index of all segments in the dataset.

        Returns:
            List[Tuple[str, float]]: A list of tuples, each containing:
                - The key of the audio file
                - The start time of the segment
        """
        index = []
        for key, metadata in self.data.items():
            if "duration" in metadata:
                duration = metadata["duration"]
            else:
                audio_path = os.path.join(self.root_dir, metadata["path"])
                info = torchaudio.info(audio_path)
                duration = info.num_frames / info.sample_rate
            num_segments = int(duration // self.segment_duration)
            for i in range(num_segments):
                segment_start_time = i * self.segment_duration
                index.append((key, segment_start_time))
        return index


class PreprocessedSegmentedDataset(Dataset):
    """A dataset class for preprocessed and segmented audio data.

    This class is designed to work with audio data that has been preprocessed and stored as feature
    vectors, along with corresponding labels.
    """

    def __init__(
        self,
        source_path: str,
        index_path: str,
        metadata_path: str,
        label_features_path: str,
        map_path: str = None,
        mode="train",
        upsample: int = 1,
    ):
        """Initialize the PreprocessedSegmentedDataset.

        Args:
            source_path (str): Path to the preprocessed audio features (npy file).
            index_path (str): Path to the index information (npy file).
            metadata_path (str): Path to the metadata JSON file.
            label_features_path (str): Path to the label features (npz file).
        """
        self.mode = mode
        self.upsample = upsample
        self.map_path = map_path
        self.source = np.load(source_path, mmap_mode="r")
        self.index = np.load(index_path, mmap_mode="r")

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.label_feature_mapper = LabelFeatureMapper.from_npz(label_features_path, map_path)

    def __len__(self) -> int:
        """Return the total number of segments in the dataset."""
        return self.source.shape[0] * self.upsample

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """Get a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Dict[str, Union[torch.Tensor, List[str]]]: A dictionary containing:
                - 'source': Preprocessed audio features
                - 'label_names': List of mood/theme labels
                - 'targets': Feature vector representation of labels
                - 'index': Index information
        """
        idx //= self.upsample
        track_id = self.index[idx]["track_id"]

        labels = self.metadata[track_id]["mood/theme"]
        label_binary = self.label_feature_mapper.get_label_binaries(labels)

        item = {
            "source": torch.from_numpy(self.source[idx]),
            "targets": self.label_feature_mapper.get_features(labels),
            "label_binary": label_binary,
        }
        if self.mode == "test":
            item.update({"label_names": labels, "index": self.index[idx]})
        return item

    def collate_fn(
        self, batch: List[Dict[str, Union[torch.Tensor, List[str]]]]
    ) -> Dict[str, Union[torch.Tensor, List[Union[str, Tuple[str, float]]]]]:
        """Collate function for creating batches.

        Args:
            batch (List[Dict[str, Union[torch.Tensor, List[str]]]]): List of individual data samples.

        Returns:
            Dict[str, Union[torch.Tensor, List[Union[str, Tuple[str, float]]]]]: A dictionary containing batched data:
                - 'source': Batched audio features
                - 'label_names': List of label names
                - 'targets': Batched target feature vectors
                - 'index': List of index information
        """
        batched_data = {
            "source": torch.stack([item["source"] for item in batch]),
            "targets": torch.stack([item["targets"] for item in batch]),
            "label_binary": torch.stack(
                [torch.from_numpy(item["label_binary"]) for item in batch]
            ),
        }
        if self.mode == "test":
            batched_data.update(
                {
                    "label_names": [item["label_names"] for item in batch],
                    "index": [item["index"] for item in batch],
                }
            )
        return batched_data


class TripletSegmentedDataset(PreprocessedSegmentedDataset):
    """A dataset class for triplet training with preprocessed and segmented audio data.

    This class outputs positive and negative targets for triplet loss training.
    """

    def __init__(
        self,
        source_path: str,
        index_path: str,
        metadata_path: str,
        label_features_path: str,
        map_path: str = None,
        combined_map_path: str = None,
        mode="train",
        num_negatives: int = 5,
        upsample: int = 1,
    ):
        """Initialize the TripletSegmentedDataset.

        Args:
            source_path (str): Path to the preprocessed audio features (npy file).
            index_path (str): Path to the index information (npy file).
            metadata_path (str): Path to the metadata JSON file.
            label_features_path (str): Path to the label features (npz file).
            mode (str): Dataset mode ('train' or 'test').
            num_negatives (int): Number of negative samples to average for the negative target.
        """
        super().__init__(
            source_path,
            index_path,
            metadata_path,
            label_features_path,
            map_path,
            mode,
            upsample=upsample,
        )
        self.num_negatives = num_negatives
        self.combined_map_path = combined_map_path
        if self.combined_map_path:
            if self.combined_map_path == "self":
                self.combined_map = self.label_feature_mapper.label2index
            else:
                self.combined_map = json.load(open(self.combined_map_path))
        else:
            self.combined_map = None

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """Get a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Dict[str, Union[torch.Tensor, List[str]]]: A dictionary containing:
                - 'source': Preprocessed audio features
                - 'label_names': List of mood/theme labels
                - 'pos_targets': Feature vector representation of positive labels
                - 'neg_targets': Feature vector representation of negative labels
                - 'index': Index information (if in test mode)
        """
        idx //= self.upsample
        item = super().__getitem__(idx)
        track_id = self.index[idx]["track_id"]
        pos_targets = item["targets"]

        # Get the indices of the positive labels
        pos_labels = self.metadata[track_id]["mood/theme"]
        pos_indices = {self.label_feature_mapper.label2index[label] for label in pos_labels}

        # Randomly pick negative labels, ensuring they are not in the positive labels
        negative_indices = []
        while len(negative_indices) < self.num_negatives:
            neg_idx = random.randint(0, len(self.label_feature_mapper.labels) - 1)
            if neg_idx not in pos_indices:
                negative_indices.append(neg_idx)
        neg_targets = torch.mean(self.label_feature_mapper.text_features[negative_indices], dim=0)

        result = {
            "source": item["source"],
            "pos_targets": pos_targets,
            "neg_targets": neg_targets,
            "label_binary": item["label_binary"],
        }
        if self.combined_map:
            result["cluster_idx"] = self.combined_map[random.choice(pos_labels)]

        if self.mode == "test":
            result.update(
                {"label_names": self.metadata[track_id]["mood/theme"], "index": self.index[idx]}
            )
        return result

    def collate_fn(
        self, batch: List[Dict[str, Union[torch.Tensor, List[str]]]]
    ) -> Dict[str, Union[torch.Tensor, List[Union[str, Dict]]]]:
        """Collate function for creating batches.

        Args:
            batch (List[Dict[str, Union[torch.Tensor, List[str]]]]): List of individual data samples.

        Returns:
            Dict[str, Union[torch.Tensor, List[Union[str, Dict]]]]: A dictionary containing batched data:
                - 'source': Batched audio features
                - 'pos_targets': Batched positive target feature vectors
                - 'neg_targets': Batched negative target feature vectors
                - 'label_names': List of label names (if in test mode)
                - 'index': List of index information (if in test mode)
        """
        batched_data = {
            "source": torch.stack([item["source"] for item in batch]),
            "pos_targets": torch.stack([item["pos_targets"] for item in batch]),
            "neg_targets": torch.stack([item["neg_targets"] for item in batch]),
            "label_binary": torch.stack(
                [torch.from_numpy(item["label_binary"]) for item in batch]
            ),
        }
        if self.combined_map:
            batched_data["cluster_idx"] = torch.tensor([item["cluster_idx"] for item in batch])

        if self.mode == "test":
            batched_data.update(
                {
                    "label_names": [item["label_names"] for item in batch],
                    "index": [item["index"] for item in batch],
                }
            )
        return batched_data

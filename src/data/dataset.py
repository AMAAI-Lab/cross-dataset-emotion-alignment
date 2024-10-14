import json
import os

import torch
import torchaudio
import torchaudio.transforms as T
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import Wav2Vec2FeatureExtractor

from util import get_label_feature_mapper


class AudioMoodDataset(Dataset):
    def __init__(
        self, data_dict, root_dir, audio_processor, segment_duration, label_feature_mapper
    ):
        self.data = data_dict
        self.root_dir = root_dir
        self.audio_processor = audio_processor
        self.sampling_rate = self.audio_processor.sampling_rate
        self.segment_duration = segment_duration
        self.keys = list(data_dict.keys())
        self.label_feature_mapper = label_feature_mapper

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data[key]

        # Get the source sample rate
        audio_path = os.path.join(self.root_dir, item["path"])
        info = torchaudio.info(audio_path)
        src_sampling_rate = info.sample_rate

        # 1. Load the audio
        audio, _ = torchaudio.load(
            audio_path, num_frames=self.segment_duration * src_sampling_rate
        )
        # 2. Resample according to the sampling rate of the processor
        if src_sampling_rate != self.sampling_rate:
            audio = T.Resample(src_sampling_rate, self.sampling_rate)(audio)

        # 3. Pad or cut the audio according to the duration
        target_length = int(self.segment_duration * self.audio_processor.sampling_rate)
        if audio.shape[1] < target_length:
            print(f"Padding audio for {audio_path}, {audio.shape[1]} -> {target_length}")
            audio = torch.nn.functional.pad(audio, (0, target_length - audio.shape[1]))
        elif audio.shape[1] > target_length:
            print(f"Cutting audio for {audio_path}, {audio.shape[1]} -> {target_length}")
            audio = audio[:, :target_length]

        # 4. Make it mono if there are multiple channels
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # 5. Go through the audio processor
        # use [0] to retrieve the mono channel
        processed_audio = self.audio_processor(
            audio, sampling_rate=self.audio_processor.sampling_rate
        )["input_values"][0]

        # 6. Text will go through the text transform
        return {
            "source": processed_audio,
            "label_names": item["mood/theme"],
            "targets": self.label_feature_mapper.get_features(item["mood/theme"]),
        }

    @classmethod
    def get_full_dataset(cls, cfg: DictConfig, device_from_sentence_transformer):
        data_dict = json.load(open(cfg.data.track_data_path))
        audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.preprocess.audio.processor)
        label_feature_mapper = get_label_feature_mapper(cfg, device_from_sentence_transformer)
        return cls(
            data_dict,
            cfg.data.audio_source_dir,
            audio_processor,
            cfg.preprocess.audio.duration,
            label_feature_mapper,
        )

    @staticmethod
    def get_default_collate_fn():
        def collate_fn(batch):
            audio = torch.stack([torch.from_numpy(item["source"]) for item in batch])
            label_names = [item["label_names"] for item in batch]
            label_features = torch.stack([item["targets"] for item in batch])
            return {"source": audio, "label_names": label_names, "targets": label_features}

        return collate_fn


def split_dataset(dataset, ratio):
    dataset_size = len(dataset)
    train_size = int(ratio * dataset_size)
    return Subset(dataset, range(train_size)), Subset(dataset, range(train_size, dataset_size))

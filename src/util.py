import concurrent.futures
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def get_label_feature_mapper(cfg, device_from_sentence_transformer, device="cpu"):
    label_feature_mapper = None
    if not os.path.exists(cfg.data.label_features_path):
        all_labels = [
            "action",
            "adventure",
            "advertising",
            "background",
            "ballad",
            "calm",
            "children",
            "christmas",
            "commercial",
            "cool",
            "corporate",
            "dark",
            "deep",
            "documentary",
            "drama",
            "dramatic",
            "dream",
            "emotional",
            "energetic",
            "epic",
            "fast",
            "film",
            "fun",
            "funny",
            "game",
            "groovy",
            "happy",
            "heavy",
            "holiday",
            "hopeful",
            "inspiring",
            "love",
            "meditative",
            "melancholic",
            "melodic",
            "motivational",
            "movie",
            "nature",
            "party",
            "positive",
            "powerful",
            "relaxing",
            "retro",
            "romantic",
            "sad",
            "sexy",
            "slow",
            "soft",
            "soundscape",
            "space",
            "sport",
            "summer",
            "trailer",
            "travel",
            "upbeat",
            "uplifting",
        ]
        # all_labels = ['action', 'adventure', 'advertising', 'ambiental', 'background', 'ballad', 'calm', 'children', 'christmas', 'commercial', 'cool', 'corporate', 'dark', 'deep', 'documentary', 'drama', 'dramatic', 'dream', 'emotional', 'energetic', 'epic', 'fast', 'film', 'fun', 'funny', 'game', 'groovy', 'happy', 'heavy', 'holiday', 'hopeful', 'horror', 'inspiring', 'love', 'meditative', 'melancholic', 'mellow', 'melodic', 'motivational', 'movie', 'nature', 'party', 'positive', 'powerful', 'relaxing', 'retro', 'romantic', 'sad', 'sexy', 'slow', 'soft', 'soundscape', 'space', 'sport', 'summer', 'trailer', 'travel', 'upbeat', 'uplifting']

        if cfg.data.name == "CAL500":
            all_labels = [
                "angry, aggressive",
                "arousing, awakening",
                "bizarre, weird",
                "calming, soothing",
                "carefree, lighthearted",
                "cheerful, festive",
                "emotional, passionate",
                "exciting, thrilling",
                "happy",
                "light, playful",
                "loving, romantic",
                "pleasant, comfortable",
                "positive, optimistic",
                "powerful, strong",
                "sad",
                "tender, soft",
                "touching, loving",
            ]
        elif cfg.data.name == "emotify":
            all_labels = [
                "amazement",
                "solemnity",
                "tenderness",
                "nostalgia",
                "calmness",
                "power",
                "joyful_activation",
                "tension",
                "sadness",
            ]
        model = SentenceTransformer(cfg.data.label_transformer).to(
            device_from_sentence_transformer
        )
        text_features = model.encode([f"{label}" for label in all_labels])
        label_feature_mapper = LabelFeatureMapper(all_labels, text_features, device)
        label_feature_mapper.save(cfg.data.label_features_path)
    else:
        label_feature_mapper = LabelFeatureMapper.from_npz(
            cfg.data.label_features_path, device_from_sentence_transformer
        )

    return label_feature_mapper


class LabelFeatureMapper:
    def __init__(self, labels, text_features, map_path=None, device="cpu"):
        self.labels = labels
        if map_path is not None:
            self.label2index = json.load(open(map_path))
        else:
            self.label2index = {label: idx for idx, label in enumerate(labels)}
        self.device = device
        self.text_features = torch.tensor(text_features).to(self.device)

    def get_label_binaries(self, labels):
        """Given a list of labels, return a binary list indicating the presence of each label in
        the mapper."""
        binary_array = np.zeros(len(self.labels), dtype=int)
        for label in labels:
            if label in self.label2index:
                binary_array[self.label2index[label]] = 1
        return binary_array

    def get_index_binaries(self, indices):
        return [1 if i in indices else 0 for i in range(len(self.labels))]

    @classmethod
    def from_npz(cls, npz_file_path, map_path=None, device="cpu"):
        data = np.load(npz_file_path, allow_pickle=True)
        labels = data["labels"]
        text_features = data["text_features"]
        return cls(labels, text_features, map_path, device)

    def get_features(self, texts, mode="mean"):
        indices = [self.label2index[text] for text in texts]
        example_features = self.text_features[indices]
        if mode == "weighted_average":
            weights = torch.rand(len(example_features), device=self.device)
            weights /= weights.sum()
            return torch.sum(example_features * weights.unsqueeze(1), dim=0)
        elif mode == "mean":
            return torch.mean(example_features, dim=0)
        elif mode == "sum":
            return torch.sum(example_features, dim=0)
        elif mode == "random":
            # randomly select one of the features
            return example_features[torch.randint(0, len(example_features), (1,))]
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def save(self, npz_file_path):
        np.savez(
            npz_file_path,
            labels=self.labels,
            text_features=self.text_features.detach().cpu().numpy(),
        )


def sim_matrix(a, b, eps=1e-8):
    """Added eps for numerical stability."""
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def create_similarity_dataframe(label_feature_mapper):
    similarity_matrix = sim_matrix(
        label_feature_mapper.text_features, label_feature_mapper.text_features
    )
    df = pd.DataFrame(similarity_matrix.cpu().numpy())

    # Set the index and column names to the labels
    df.index = label_feature_mapper.labels
    df.columns = df.index

    return df


def convert_mp3_to_wav_pydub(src_path, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if not os.path.exists(target_path):
        audio = AudioSegment.from_mp3(src_path)
        audio.export(target_path, format="wav")


def convert_mp3_to_wav_ffmpeg(src_path, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if not os.path.exists(target_path):
        subprocess.run(["ffmpeg", "-i", src_path, target_path], check=True, capture_output=True)


def process_mp3_files(src_folder, target_folder, use_ffmpeg=False):
    mp3_files = []
    for root, _, files in os.walk(src_folder):
        for file in files:
            if file.lower().endswith(".mp3"):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, src_folder)
                target_path = os.path.join(target_folder, os.path.splitext(rel_path)[0] + ".wav")
                mp3_files.append((src_path, target_path))

    convert_func = convert_mp3_to_wav_ffmpeg if use_ffmpeg else convert_mp3_to_wav_pydub

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(convert_func, src, target) for src, target in mp3_files]
        for _ in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(mp3_files),
            desc="Converting MP3 to WAV",
        ):
            pass


def convert_track_metadata_to_wav(metadata_path, metadata_wav_path):
    with open(metadata_path) as file:
        data = json.load(file)

    # for each track in the data, change the path from mp3 to wav
    for track in data.values():
        track["path"] = track["path"][:-4] + ".wav"

    # write the data back to the json file
    with open(metadata_wav_path, "w") as file:
        json.dump(data, file)

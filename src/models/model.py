from typing import Any, Dict, List, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import F1Score, HammingDistance, Precision, Recall

from src.models.components.regularization_loss import (
    RegularizationLoss,
    RegularizationLossV2,
    RegularizationLossV3,
)
from src.models.components.song_level_metric import SongLevelMetric
from util import LabelFeatureMapper


class BaseModule(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        datamodule: pl.LightningDataModule,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["datamodule"])
        self.datamodule = datamodule
        self.net = net
        self.dataloader_map = {
            "mtg": 0,
            "CAL500": 1,
            "emotify": 2,
        }

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            monitor_metric = f"val_f1/data{self.dataloader_map[self.datamodule.hparams.combine_train_datasets[0]]}"
            print("MONITOR METRIC:", monitor_metric)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor_metric,
                    "frequency": 1,
                    "interval": "epoch",
                },
            }
        return {"optimizer": optimizer}


class CombinedMSEModule(BaseModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        datamodule: pl.LightningDataModule,
        dataset_label_paths: List[str],
        top_k_list: List[int],
        compile: bool,
    ) -> None:
        super().__init__(net, optimizer, scheduler, datamodule, compile)
        self.label_mappers = [LabelFeatureMapper.from_npz(path) for path in dataset_label_paths]
        self.loss_fn = nn.MSELoss()
        self.top_k_list = top_k_list

        # Initialize metrics for each dataset
        self.metrics = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "precision": Precision(
                            task="multilabel", num_labels=len(mapper.labels), average="macro"
                        ),
                        "recall": Recall(
                            task="multilabel", num_labels=len(mapper.labels), average="macro"
                        ),
                        "f1_score": F1Score(
                            task="multilabel", num_labels=len(mapper.labels), average="macro"
                        ),
                    }
                )
                for mapper in self.label_mappers
            ]
        )

    def training_step(self, batch_dict, batch_idx):
        total_loss = 0
        num_datasets = len(batch_dict)

        for key, batch in batch_dict.items():
            if batch is None:
                continue
            source = batch["source"].squeeze(1)
            targets = batch["targets"]
            outputs = self.net(source)
            loss = self.loss_fn(outputs, targets)
            total_loss += loss

            # Log individual dataset loss
            self.log(f"train_loss_{key}", loss)

        # Calculate and log average loss
        avg_loss = total_loss / num_datasets
        self.log("train_loss", avg_loss, on_step=True)

        return avg_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        source = batch["source"].squeeze(1)
        targets = batch["targets"]
        label_binary = batch["label_binary"]
        outputs = self.net(source)
        loss = self.loss_fn(outputs, targets)
        self.log("val_loss", loss, prog_bar=True)

        # Calculate metrics
        similarities = F.cosine_similarity(
            outputs.unsqueeze(1),
            self.label_mappers[dataloader_idx].text_features.unsqueeze(0).to(self.device),
            dim=-1,
        )
        _, top_k_indices = torch.topk(similarities, k=self.top_k_list[dataloader_idx], dim=1)
        predicted_labels = torch.zeros_like(similarities).scatter_(1, top_k_indices, 1)

        self.metrics[dataloader_idx]["precision"].update(predicted_labels, label_binary)
        self.metrics[dataloader_idx]["recall"].update(predicted_labels, label_binary)
        self.metrics[dataloader_idx]["f1_score"].update(predicted_labels, label_binary)

        return loss

    def on_validation_epoch_end(self):
        for idx, metric_dict in enumerate(self.metrics):
            precision = metric_dict["precision"].compute()
            recall = metric_dict["recall"].compute()
            f1_score = metric_dict["f1_score"].compute()

            self.log(f"val_precision/data{idx}", precision)
            self.log(f"val_recall/data{idx}", recall)
            self.log(f"val_f1/data{idx}", f1_score)

            metric_dict["precision"].reset()
            metric_dict["recall"].reset()
            metric_dict["f1_score"].reset()


class CombinedTripletModule(BaseModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        datamodule: pl.LightningDataModule,
        dataset_label_paths: List[str],
        top_k_list: List[int],
        compile: bool = False,
        normalize: bool = True,
        regularization_alpha: Union[float, None] = None,
        regularization_version: str = "v2",
        margin: float = 1.0,
    ) -> None:
        super().__init__(net, optimizer, scheduler, datamodule, compile)
        self.hparams.normalize = normalize
        self.hparams.regularization_alpha = regularization_alpha

        if regularization_alpha is not None:
            if regularization_version == "v2":
                self.reg_loss = RegularizationLossV2(regularization_alpha)
            elif regularization_version == "v3":
                self.reg_loss = RegularizationLossV3(regularization_alpha)
            else:
                self.reg_loss = RegularizationLoss(regularization_alpha)
        else:
            self.reg_loss = None

        self.hparams.margin = margin
        self.label_mappers = [LabelFeatureMapper.from_npz(path) for path in dataset_label_paths]
        self.top_k_list = top_k_list

        # Initialize metrics for each dataset
        self.metrics = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "precision": Precision(
                            task="multilabel", num_labels=len(mapper.labels), average="macro"
                        ),
                        "recall": Recall(
                            task="multilabel", num_labels=len(mapper.labels), average="macro"
                        ),
                        "f1_score": F1Score(
                            task="multilabel", num_labels=len(mapper.labels), average="macro"
                        ),
                        "hamming_loss": HammingDistance(
                            task="multilabel", num_labels=len(mapper.labels)
                        ),
                    }
                )
                for mapper in self.label_mappers
            ]
        )

        # Initialize song-level metrics for each dataset
        self.song_level_metrics = nn.ModuleList(
            [
                SongLevelMetric(num_labels=len(mapper.labels), top_k=top_k)
                for mapper, top_k in zip(self.label_mappers, self.top_k_list)
            ]
        )

        # Initialize track_id to integer id map
        self.track_id_map = {}
        self.next_id = 0

    def triplet_loss_fn(self, anchor, positive, negative):
        pos_dist = F.cosine_similarity(anchor, positive)
        neg_dist = F.cosine_similarity(anchor, negative)
        loss = F.relu(neg_dist - pos_dist + self.hparams.margin)
        return loss.mean()

    def similarity_loss_fn(self, anchor, positive):
        return 1 - F.cosine_similarity(anchor, positive).mean()

    def training_step(self, batch_dict, batch_idx):
        total_loss = 0
        num_datasets = len(batch_dict)
        anchors = []
        cluster_centers = []

        for key, batch in batch_dict.items():
            if batch is None:
                continue
            source = batch["source"].squeeze(1)
            pos_targets = batch["pos_targets"]
            neg_targets = batch["neg_targets"]

            anchor = self.net(source)
            if self.hparams.normalize:
                anchor = F.normalize(anchor, p=2, dim=-1)

            if self.reg_loss is not None:
                anchors.append(anchor)
                cluster_centers.append(batch["cluster_idx"])

            loss = self.triplet_loss_fn(anchor, pos_targets, neg_targets)
            total_loss += loss

            # Log individual dataset loss
            self.log(f"train_loss_{key}", loss)

        if self.reg_loss is not None:
            anchors = torch.cat(anchors, dim=0)
            cluster_centers = torch.cat(cluster_centers, dim=0)
            reg_loss = self.reg_loss(anchors, cluster_centers)
            total_loss += reg_loss
            self.log("train_reg_loss", reg_loss)

        self.log("train_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        source = batch["source"].squeeze(1)
        pos_targets = batch["pos_targets"]
        neg_targets = batch["neg_targets"]
        label_binary = batch["label_binary"]
        anchor = self.net(source)
        pos_loss = self.similarity_loss_fn(anchor, pos_targets)
        self.log("val_loss_pos", pos_loss, prog_bar=True)

        triplet_loss = self.triplet_loss_fn(anchor, pos_targets, neg_targets)
        self.log("val_loss_triplet", triplet_loss, prog_bar=True)

        # Calculate metrics
        similarities = F.cosine_similarity(
            anchor.unsqueeze(1),
            self.label_mappers[dataloader_idx].text_features.unsqueeze(0).to(self.device),
            dim=-1,
        )
        _, top_k_indices = torch.topk(similarities, k=self.top_k_list[dataloader_idx], dim=1)
        predicted_labels = torch.zeros_like(similarities, device=self.device).scatter_(
            1, top_k_indices, 1
        )
        self.metrics[dataloader_idx]["precision"].update(predicted_labels, label_binary)
        self.metrics[dataloader_idx]["recall"].update(predicted_labels, label_binary)
        self.metrics[dataloader_idx]["f1_score"].update(predicted_labels, label_binary)
        self.metrics[dataloader_idx]["hamming_loss"].update(predicted_labels, label_binary)

        return triplet_loss

    def on_validation_epoch_end(self):
        for idx, metric_dict in enumerate(self.metrics):
            precision = metric_dict["precision"].compute()
            recall = metric_dict["recall"].compute()
            f1_score = metric_dict["f1_score"].compute()
            hamming_loss = metric_dict["hamming_loss"].compute()

            self.log(f"val_precision/data{idx}", precision)
            self.log(f"val_recall/data{idx}", recall)
            self.log(f"val_f1/data{idx}", f1_score)
            self.log(f"val_hamming_loss/data{idx}", hamming_loss)

            metric_dict["precision"].reset()
            metric_dict["recall"].reset()
            metric_dict["f1_score"].reset()
            metric_dict["hamming_loss"].reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        source = batch["source"].squeeze(1)
        label_binary = batch["label_binary"]
        track_ids = [idx["track_id"] for idx in batch["index"]]

        # Convert track_ids to integer ids
        int_track_ids = []
        for track_id in track_ids:
            if track_id not in self.track_id_map:
                self.track_id_map[track_id] = self.next_id
                self.next_id += 1
            int_track_ids.append(self.track_id_map[track_id])

        int_track_ids = torch.tensor(int_track_ids)

        outputs = self.net(source)

        # Calculate similarities
        similarities = F.cosine_similarity(
            outputs.unsqueeze(1),
            self.label_mappers[dataloader_idx].text_features.unsqueeze(0).to(self.device),
            dim=-1,
        )

        # Update song-level metric
        self.song_level_metrics[dataloader_idx].update(similarities, int_track_ids, label_binary)

    def on_test_epoch_end(self):
        for idx, metric in enumerate(self.song_level_metrics):
            results = metric.compute()
            self.log(f"test_song_precision/data{idx}", results["precision"])
            self.log(f"test_song_recall/data{idx}", results["recall"])
            self.log(f"test_song_f1/data{idx}", results["f1_score"])
            metric.reset()


class MusicMoodModel(nn.Module):
    def __init__(self, d_model, output_size, num_layers=3, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.selected_layers = [3, 6, 9, 12]

        layers = []
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(
                        self.d_model * len(self.selected_layers),
                        self.d_model * len(self.selected_layers),
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )

        layers.append(nn.Linear(self.d_model * len(self.selected_layers), output_size))

        self.sequential = nn.Sequential(*layers)

    def forward(self, source):
        # Extract and concatenate layers 3, 6, 9, 12
        x = source[:, self.selected_layers, :].view(-1, len(self.selected_layers) * self.d_model)
        return self.sequential(x)


class MusicMoodModelLayer6(nn.Module):
    def __init__(self, d_model, output_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.selected_layers = [6]

        self.sequential = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.d_model * 4, 4 * output_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * output_size, output_size),
        )

    def forward(self, source):
        # Extract and concatenate layers 3, 6, 9, 12
        x = source[:, self.selected_layers, :].view(-1, len(self.selected_layers) * self.d_model)
        return self.sequential(x)


class MusicMoodModelWithAttention(nn.Module):
    def __init__(self, d_model, output_size, nhead=8, num_layers=3, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.selected_layers = [3, 6, 9, 12]

        # Input projection
        self.input_proj = nn.Linear(4 * self.d_model, self.d_model)

        # Self-attention blocks
        self.attention_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=nhead,
                    dim_feedforward=4 * self.d_model,
                    dropout=dropout_rate,
                    activation="relu",
                )
                for _ in range(num_layers)
            ]
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.d_model)

        # Output projection
        self.output_proj = nn.Linear(self.d_model, output_size)

    def forward(self, source):
        # Extract and concatenate layers 3, 6, 9, 12
        x = source[:, self.selected_layers, :].view(-1, len(self.selected_layers) * self.d_model)

        # Project input to d_model dimensions
        x = self.input_proj(x)

        # Reshape for self-attention (seq_len=1 as we're dealing with a single "token" per sample)
        x = x.unsqueeze(0)  # Shape: (1, batch_size, d_model)

        # Apply self-attention blocks
        for block in self.attention_blocks:
            x = block(x)

        # Apply layer normalization
        x = self.layer_norm(x)

        # Reshape and project to output size
        x = x.squeeze(0)  # Shape: (batch_size, d_model)
        output = self.output_proj(x)

        return output

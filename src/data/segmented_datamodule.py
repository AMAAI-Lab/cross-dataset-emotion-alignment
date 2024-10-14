from typing import Any, Callable, Dict, List, Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.data.segmented_dataset import (
    PreprocessedSegmentedDataset,
    TripletSegmentedDataset,
)


class BaseSegmentedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: any,
        val_dataset: any,
        test_dataset: any,
        batch_size: int,
        num_workers: int,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_dataset", "val_dataset", "test_dataset"])
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def _create_dataloader(self, dataset, shuffle, collate_fn):
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        return self._create_dataloader(
            self.train_dataset, shuffle=True, collate_fn=self.train_dataset.collate_fn
        )

    def val_dataloader(self):
        return self._create_dataloader(
            self.val_dataset, shuffle=False, collate_fn=self.val_dataset.collate_fn
        )

    def test_dataloader(self):
        return self._create_dataloader(
            self.test_dataset, shuffle=False, collate_fn=self.test_dataset.collate_fn
        )


class SplitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: any,
        batch_size: int,
        num_workers: int,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["dataset"])
        total_size = len(dataset)
        train_size = int(self.hparams.train_ratio * total_size)
        val_size = int(self.hparams.val_ratio * total_size)
        train_indices = range(train_size)
        val_indices = range(train_size, train_size + val_size)
        test_indices = range(train_size + val_size, total_size)

        self.dataset = dataset
        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)
        self.test_dataset = Subset(self.dataset, test_indices)
        self.collate_fn = self.dataset.collate_fn

    def _create_dataloader(self, dataset, shuffle, collate_fn):
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        return self._create_dataloader(
            self.train_dataset, shuffle=True, collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return self._create_dataloader(
            self.test_dataset, shuffle=False, collate_fn=self.dataset.collate_fn
        )


class CombinedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets_info: Dict[str, any],
        train_combine_mode: str,
        combine_train_datasets: List[str],
        batch_size: int,
        num_workers: int,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        train_split_ratio: float = 0.8,
        val_split_ratio: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["datasets_info"])
        self.datasets_info = datasets_info

    def setup(self, stage: Optional[str] = None):
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []

        for dataset_name, dataset in self.datasets_info.items():
            if dataset_name == "mtg":
                train_dataset = dataset.train
                val_dataset = dataset.val
                test_dataset = dataset.test
                train_collate_fn = train_dataset.collate_fn
                val_collate_fn = val_dataset.collate_fn
                test_collate_fn = test_dataset.collate_fn
            else:
                total_size = len(dataset)
                train_size = int(self.hparams.train_split_ratio * total_size)
                val_size = int(self.hparams.val_split_ratio * total_size)
                train_collate_fn = dataset.collate_fn
                val_collate_fn = dataset.collate_fn
                test_collate_fn = dataset.collate_fn

                train_dataset = torch.utils.data.Subset(dataset, range(train_size))
                val_dataset = torch.utils.data.Subset(
                    dataset, range(train_size, train_size + val_size)
                )
                test_dataset = torch.utils.data.Subset(
                    dataset, range(train_size + val_size, total_size)
                )

            self.train_datasets.append((dataset_name, train_dataset, train_collate_fn))
            self.val_datasets.append((dataset_name, val_dataset, val_collate_fn))
            self.test_datasets.append((dataset_name, test_dataset, test_collate_fn))

    def _create_dataloader(self, dataset, shuffle, collate_fn):
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        if self.hparams.combine_train_datasets:
            combined_dataloaders = {}
            for dataset_name, dataset, collate_fn in self.train_datasets:
                if dataset_name in self.hparams.combine_train_datasets:
                    combined_dataloaders[dataset_name] = self._create_dataloader(
                        dataset, shuffle=True, collate_fn=collate_fn
                    )
            combined_dataset = CombinedLoader(
                combined_dataloaders, mode=self.hparams.train_combine_mode
            )
            return combined_dataset
        else:
            return [
                self._create_dataloader(dataset, shuffle=True, collate_fn=collate_fn)
                for _, dataset, collate_fn in self.train_datasets
            ]

    def val_dataloader(self):
        return [
            self._create_dataloader(dataset, shuffle=False, collate_fn=collate_fn)
            for _, dataset, collate_fn in self.val_datasets
        ]

    def test_dataloader(self):
        return [
            self._create_dataloader(dataset, shuffle=False, collate_fn=collate_fn)
            for _, dataset, collate_fn in self.test_datasets
        ]

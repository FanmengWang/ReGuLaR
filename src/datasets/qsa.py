if __name__ == "__main__":
    import sys

    sys.path.append("../../")

import os
import copy
from pathlib import Path
import json
import torch
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl


def steps_collate_fn(batch):
    keys = batch[0].keys()
    output = {}
    for key in keys:
        if key == "steps":
            output[key] = [item[key] for item in batch]
        else:
            items = [item[key] for item in batch]
            try:
                output[key] = torch.utils.data._utils.collate.default_collate(items)
            except:
                output[key] = items
    return output


def get_precompute_path(base_name, data):
    try:
        idx = data.get("image_idx")
        if not idx:
            return ""
        return os.path.join(base_name, idx.split('_')[0], f'{idx}.pt')
    except Exception:
        return ""


class QuestionStepsAnswerDataset(Dataset):
    def __init__(
        self,
        data: List[dict],
        precompute_name: str,
    ):
        super().__init__()
        self.data = {}
        self.precompute_name = precompute_name
        for idx, d in enumerate(data):
            self.data[idx] = {
                "idx": idx,
                "question": d["question"],
                "answer": d["answer"],
                "steps": d["steps"],
                "n_steps": len(d["steps"]),
                "precompute_file": get_precompute_path(self.precompute_name, d)
            }
        self.all_indices = list(self.data.keys())
        self.indices = copy.deepcopy(self.all_indices)

    def get_all_indices(self):
        return self.all_indices

    def set_indices(self, indices: List[int]):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        data_idx = self.indices[idx]
        return self.data[data_idx]


class QSADataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, precompute_name, tiny_dataset=False, epoch_scaling=1, all_config=None):
        super().__init__()
        self.dataset_name = dataset_name
        self.precompute_name = precompute_name
        self.dataset_dir = Path(all_config.args.workspace_path, "datasets", dataset_name)
        self.tiny_dataset = tiny_dataset
        self.epoch_scaling = epoch_scaling
        self.all_config = all_config
        self.batch_size = all_config.dataloader.batch_size
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage: str = None):
        def load_split(split: str):
            with open(self.dataset_dir / f"{split}.json") as f:
                data = json.load(f)
                if self.tiny_dataset:
                    data = data[:32]
            return data
        if stage == "fit":
            self.train_set = self._create_dataset(load_split("train"), "train")
            self.val_set = self._create_dataset(load_split("val"), "val")
        elif stage == "test":
            self.test_set = self._create_dataset(load_split("test"), "test")

    def _create_dataset(self, raw_data: List[dict], mode: str) -> QuestionStepsAnswerDataset:
        return QuestionStepsAnswerDataset(
            data=raw_data,
            precompute_name=self.precompute_name,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.all_config.dataloader.batch_size,
            num_workers=self.all_config.dataloader.get("num_workers", 4),
            pin_memory=self.all_config.dataloader.get("pin_memory", True),
            persistent_workers=self.all_config.dataloader.get("persistent_workers", True),
            collate_fn=steps_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.all_config.dataloader.get("val_batch_size", 1),
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=steps_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.all_config.dataloader.get("val_batch_size", 1),
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=steps_collate_fn,
        )

    def get_dataloader_to_filter_indices(self):
        return DataLoader(
            self.train_set,
            batch_size=8,
            shuffle=False,
            collate_fn=steps_collate_fn,
        )

    def get_all_train_indices(self):
        return self.train_set.get_all_indices()

    def set_train_indices(self, train_indices):
        self.train_set.set_indices(train_indices)

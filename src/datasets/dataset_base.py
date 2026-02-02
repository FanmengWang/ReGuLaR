from pathlib import Path
import copy
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from ..utils.utils import instantiate_from_config


class DatasetBase(Dataset):
    def __init__(self, dataset_dir, split="train", epoch_scaling=1, tiny_dataset=False):
        self.dataset_dir = Path(dataset_dir).expanduser()
        self.split = split
        self.epoch_scaling = epoch_scaling
        self.tiny_dataset = tiny_dataset 

    @property
    def real_length(self):
        raise NotImplementedError

    def __len__(self):
        if self.split == "train":
            return self.real_length * self.epoch_scaling
        else:
            return self.real_length

    def __getitem__(self, index):
        index = index % self.real_length
        return self.getitem(index=index)

    def getitem(self, index):
        raise NotImplementedError("Should be implemented in the child class.")


class DataModuleBase(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name,
        tiny_dataset=False,
        epoch_scaling=1,
        all_config=None,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.tiny_dataset = tiny_dataset
        self.epoch_scaling = epoch_scaling
        self.all_config = all_config

    def get_dataloaders(self, stage):
        if stage == "fit":
            train_ds = instantiate_from_config(self.all_config.dataset, extra_kwargs={"split": "train"})
            val_ds = instantiate_from_config(self.all_config.dataset, extra_kwargs={"split": "val"})

            dataloader_config = copy.deepcopy(self.all_config.dataloader)

            train_dataloader = DataLoader(train_ds, **dataloader_config, shuffle=True, drop_last=True)
            val_batch_size = dataloader_config.pop("val_batch_size", dataloader_config.batch_size)
            dataloader_config.batch_size = val_batch_size
            val_dataloader = DataLoader(val_ds, **dataloader_config, shuffle=False, drop_last=True)
            return train_dataloader, val_dataloader

        elif stage == "test":
            dataloader_config = copy.deepcopy(self.all_config.dataloader)
            val_batch_size = dataloader_config.pop("val_batch_size", dataloader_config.batch_size)
            dataloader_config.batch_size = val_batch_size
            test_ds = instantiate_from_config(self.all_config.dataset, extra_kwargs={"split": "test"})
            test_dataloader = DataLoader(test_ds, **dataloader_config, shuffle=False, drop_last=True)
            return test_dataloader

    def setup(self, stage):
        if stage == "fit":
            self._train_dataloader, self._val_dataloader = self.get_dataloaders(stage="fit")
            epoch_length = len(self.train_dataloader()) // len(self.all_config.trainer.devices)
            self.all_config.model.training_kwargs["num_training_steps"] = (
                epoch_length * self.all_config.trainer.max_epochs
            )
        elif stage == "test":
            self._test_dataloader = self.get_dataloaders(stage="test")

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

from .utils import get_timestamp
import json
import logging
from pathlib import Path
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only


class JsonLogger:
    def __init__(self, pl_class: pl.LightningModule, save_dir=None, log_file_name: str = "train", tmp_log=False):
        self.log_data = dict()

        if tmp_log:
            self.log_path = Path("temp_log.json")
            return

        try:
            if save_dir is None:
                save_dir = pl_class.logger.log_dir
            self.log_path = Path(save_dir) / f"{log_file_name}.json"
        except AttributeError:
            self.log_path = Path("temp_log.json")

    @rank_zero_only
    def log(self, message: dict):
        self.log_data.update(message)

        json_message = json.dumps(self.log_data, indent=2)
        with self.log_path.open("w") as f:
            f.write(json_message + "\n")


class TextLogger:
    def __init__(self, pl_class: pl.LightningModule, save_dir=None, log_file_name: str = "train", tmp_log=False):
        if tmp_log:
            self.log_path = Path("temp_log.txt")
            return

        try:
            if save_dir is None:
                save_dir = pl_class.logger.log_dir
            self.log_path = Path(save_dir) / f"{log_file_name}.txt"
        except AttributeError:
            self.log_path = Path("temp_log.txt")

    @rank_zero_only
    def log(self, message: str):
        msg = f"{get_timestamp()}:\n{message}\n"
        with self.log_path.open("a") as f:
            print(msg)
            f.write(msg)


def setup_logger(name, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    logger.addHandler(console_handler)
    if log_file:
        logger.addHandler(file_handler)

    return logger

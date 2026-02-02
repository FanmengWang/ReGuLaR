import sys
from typing import Dict, Any
import copy
import importlib
from pathlib import Path
from datetime import datetime
from omegaconf.dictconfig import DictConfig
import numpy as np
import torch


def sample_indices_from_attention_mask_3d(attention_mask_3d: torch.Tensor) -> torch.Tensor:
    batch_size, seq_length, r = attention_mask_3d.shape
    x_flat = attention_mask_3d.view(-1, r) + 1e-5  # Flatten to (batch_size * seq_length, r), avoid [0, 0, 0, 0]

    sum_ones = x_flat.sum(dim=1, keepdim=True)
    probs_flat = x_flat / sum_ones  # Normalize to get probabilities

    # Sample one index per flattened position
    indices_flat = torch.multinomial(probs_flat, num_samples=1)  # (batch_size * seq_length, 1)

    return indices_flat.view(batch_size, seq_length, 1)


def batch_masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # x: [B, T]
    # mask: [B, T]
    # return: []
    return torch.sum(x * mask) / mask.sum()


def get_position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    # attention_mask: [B, T]
    # position_ids: [B, T]
    position_ids = torch.clamp_min(torch.cumsum(attention_mask, dim=1) - 1, 0)
    return position_ids


def swap(a, b):
    return b, a


def get_timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, extra_kwargs=dict()):
    config_dict = dict(config)
    if "target" not in config_dict:
        raise ValueError(f"target not found in {config}")

    target_kwargs = copy.deepcopy(config_dict)
    target_kwargs.pop("target")

    for k, v in target_kwargs.items():
        if isinstance(v, DictConfig) and "target" in v.keys():
            target_kwargs[k] = instantiate_from_config(v)
    target_kwargs.update(extra_kwargs)

    return get_obj_from_str(config_dict["target"])(**target_kwargs)


def dict_apply(x, func):
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def dict_to_device(d: Dict[str, Any], device):
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(device=device)
    return d


def list_subdirs(path: Path):
    return [d for d in path.glob("*") if not d.is_file()]


def is_debug_mode():
    return hasattr(sys, "gettrace") and sys.gettrace() is not None


def get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval

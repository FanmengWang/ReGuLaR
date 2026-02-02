from typing import Optional
from dataclasses import dataclass, fields
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Experience:
    latent_logprobs: torch.Tensor = None
    answer_logprobs: torch.Tensor = None

    question_input_ids: torch.Tensor = None
    question_attention_mask: torch.Tensor = None
    latent_inputs_embeds: torch.Tensor = None
    latent_attention_mask: torch.Tensor = None
    answer_input_ids: torch.Tensor = None
    answer_attention_mask: torch.Tensor = None

    n_latent_forward: torch.Tensor = None

    rewards: Optional[torch.Tensor] = None
    accuracies: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None

    def to(self, device: torch.device):
        members = {}
        for field in fields(self):
            v = getattr(self, field.name)
            if isinstance(v, torch.Tensor):
                v = v.to(device=device)
            members[field.name] = v
        return Experience(**members)


keys = [field.name for field in fields(Experience)]


class ReplayBuffer:
    def __init__(self, limit: int = 0) -> None:
        self.limit = limit
        self.items: list[Experience] = []

    def append(self, experience: Experience) -> None:
        items = split_experience_batch(experience)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Experience:
        return self.items[idx]


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def join_experience_batch(items: list[Experience]) -> Experience:
    batch_data = {}
    left_pad_keys = ("question_input_ids", "question_attention_mask", "question_position_ids")
    for key in keys:
        vals = [getattr(item, key) for item in items]
        if all(v is not None for v in vals):
            if key in left_pad_keys:
                data = zero_pad_sequences(vals, "left")
            else:
                data = zero_pad_sequences(vals, "right")
        else:
            data = None
        batch_data[key] = data
    return Experience(**batch_data)


def split_experience_batch(experience: Experience) -> list[Experience]:
    batch_size = experience.question_input_ids.size(0)
    batch_data = [{} for _ in range(batch_size)]

    for key in keys:
        value = getattr(experience, key)
        if value is None:
            vals = [None] * batch_size
        else:
            vals = torch.unbind(value)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_data[i][key] = v

    return [Experience(**data) for data in batch_data]


def zero_pad_sequences(sequences: list[torch.Tensor], side: str = "left") -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        if len(seq.shape) == 1:
            padding = (pad_len, 0) if side == "left" else (0, pad_len)
        else:
            padding = (0, 0, pad_len, 0) if side == "left" else (0, 0, 0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


def masked_sum(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
) -> torch.Tensor:
    return (tensor * mask).sum(axis=dim) / 32


class GRPOLoss(nn.Module):
    """GRPO actor loss"""

    def __init__(self, rl_config) -> None:
        super().__init__()
        self.clip_eps = rl_config.clip_eps
        self.use_latent_loss = rl_config.use_latent_loss
        self.use_answer_loss = rl_config.use_answer_loss
        self.average_per_token_loss = rl_config.get("average_per_token_loss", True)

    def calculate_loss(self, logprobs, logprobs_old, attention_mask, advantages):
        ratio = (logprobs - logprobs_old).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        if self.average_per_token_loss:
            loss = masked_mean(loss, attention_mask, dim=-1).mean()
        else:
            loss = masked_sum(loss, attention_mask, dim=-1).mean()
        return loss

    def forward(
        self,
        latent_logprobs,
        answer_logprobs: torch.Tensor,
        experience: Experience,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_latent_loss:
            latent_loss = self.calculate_loss(
                latent_logprobs,
                experience.latent_logprobs,
                experience.latent_attention_mask,
                experience.advantages,
            )
        else:
            latent_loss = 0

        if self.use_answer_loss:
            answer_loss = self.calculate_loss(
                answer_logprobs,
                experience.answer_logprobs,
                experience.answer_attention_mask,
                experience.advantages,
            )
        else:
            answer_loss = 0

        return {
            "total_loss": latent_loss + answer_loss,
            "latent_loss": latent_loss,
            "answer_loss": answer_loss,
        }

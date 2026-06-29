import math
from typing import Literal, Optional


def patch_tokens_per_image(crop_size: int, patch_size: int) -> int:
    """Number of patch tokens per image (excludes class token)."""
    if crop_size % patch_size != 0:
        raise ValueError(f"crop_size {crop_size} must be divisible by patch_size {patch_size}")
    return (crop_size // patch_size) ** 2


def token_matched_batch_size(
    crop_size: int,
    reference_crop_size: int,
    reference_batch_size: int,
) -> int:
    """Batch size that keeps total patch tokens per step constant."""
    scale = (reference_crop_size / crop_size) ** 2
    return max(1, round(reference_batch_size * scale))


def token_matched_learning_rate(
    crop_size: int,
    reference_crop_size: int,
    reference_lr: float,
    reference_batch_size: int,
) -> float:
    """Learning rate scaled linearly with the token-matched batch size."""
    batch_size = token_matched_batch_size(
        crop_size, reference_crop_size, reference_batch_size
    )
    return reference_lr * (batch_size / reference_batch_size)


def build_crop_schedule(
    start_crop_size: int,
    full_crop_size: int,
    step_size: int,
) -> list[int]:
    sizes = list(range(start_crop_size, full_crop_size + 1, step_size))
    if not sizes:
        return [full_crop_size]
    if sizes[-1] != full_crop_size:
        sizes.append(full_crop_size)
    return sizes


def crop_size_for_epoch(
    epoch: int,
    start_crop_size: int,
    full_crop_size: int,
    mode: Literal["linear", "direct"],
    step_size: int,
    epochs_per_stage: int,
    switch_epoch: Optional[int],
    schedule: Optional[list[int]] = None,
) -> int:
    if mode == "direct":
        if switch_epoch is None:
            raise ValueError("switch_epoch is required when mode='direct'")
        if epoch < switch_epoch:
            return start_crop_size
        return full_crop_size

    if schedule is None:
        schedule = build_crop_schedule(start_crop_size, full_crop_size, step_size)
    stage = min(epoch // epochs_per_stage, len(schedule) - 1)
    return schedule[stage]


def estimate_training_steps(
    num_train_samples: int,
    max_epochs: int,
    start_crop_size: int,
    full_crop_size: int,
    mode: Literal["linear", "direct"],
    step_size: int,
    epochs_per_stage: int,
    switch_epoch: Optional[int],
    reference_crop_size: int,
    reference_batch_size: int,
) -> int:
    """Total training steps when batch size is token-matched each epoch."""
    schedule = None
    if mode == "linear":
        schedule = build_crop_schedule(start_crop_size, full_crop_size, step_size)

    total_steps = 0
    for epoch in range(max_epochs):
        crop_size = crop_size_for_epoch(
            epoch,
            start_crop_size,
            full_crop_size,
            mode,
            step_size,
            epochs_per_stage,
            switch_epoch,
            schedule,
        )
        batch_size = token_matched_batch_size(
            crop_size, reference_crop_size, reference_batch_size
        )
        total_steps += math.ceil(num_train_samples / batch_size)
    return total_steps

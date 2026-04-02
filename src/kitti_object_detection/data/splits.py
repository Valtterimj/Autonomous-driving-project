import random
from typing import Sequence, TypeVar

T = TypeVar("T")

def train_val_split(
    items: Sequence[T],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[T], list[T]]:
    """
    Randomly split items into train and validation sets using fixed seed.
    Returns:
        train_items, val_items
    """
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)

    num_val = int(len(items) * val_fraction)
    val_items = items[:num_val]
    train_items = items[num_val:]

    return train_items, val_items

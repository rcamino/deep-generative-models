import random
import torch

import numpy as np

from typing import Optional


def seed_all(seed: Optional[int]) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

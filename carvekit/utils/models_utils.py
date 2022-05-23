"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""

import random
import warnings
import torch


def fix_seed(seed=42):
    """Sets fixed random seed

    Args:
        seed: Random seed to be set
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False
    return True


def suppress_warnings():
    # Suppress PyTorch 1.11.0 warning associated with changing order of args in nn.MaxPool2d layer,
    # since source code is not affected by this issue and there aren't any other correct way to hide this message.
    warnings.filterwarnings("ignore",
                            category=UserWarning,
                            message="Note that order of the arguments: ceil_mode and "
                                    "return_indices will changeto match the args list "
                                    "in nn.MaxPool2d in a future release.",
                            module="torch")

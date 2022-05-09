"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
from carvekit.utils.pool_utils import batch_generator, thread_pool_processing


def test_thread_pool_processing():
    assert thread_pool_processing(int, ["1", "2", "3"]) == [1, 2, 3]
    assert thread_pool_processing(int, ["1", "2", "3"], workers=1) == [1, 2, 3]


def test_batch_generator():
    assert list(batch_generator([1, 2, 3], n=1)) == [[1], [2], [3]]
    assert list(batch_generator([1, 2, 3, 4], n=2)) == [[1, 2], [3, 4]]
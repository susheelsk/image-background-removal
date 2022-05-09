"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable


def thread_pool_processing(func: Any, data: Iterable, workers=18):
    """
        Passes all iterator data through the given function

        Args:
            workers: Count of workers.
            func: function to pass data through
            data: input iterator

        Returns:
            function return list

    """
    with ThreadPoolExecutor(workers) as p:
        return list(p.map(func, data))


def batch_generator(iterable, n=1):
    """
        Splits any iterable into n-size packets

        Args:
            iterable: iterator
            n: size of packets

        Returns:
            new n-size packet
    """
    it = len(iterable)
    for ndx in range(0, it, n):
        yield iterable[ndx:min(ndx + n, it)]

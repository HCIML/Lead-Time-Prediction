import sys
from functools import wraps
import pickle as pcl
from tqdm import trange
import logging
import os
import pathlib
from parameters import cache_location

'''
USAGE:
@cache_to_disk
def some_func() with arbitrary input and return arg. Input must be in the format: fct(i1=i1, i2=i2)
'''
def cache_to_disk(func):
    """
    Caches a function's result to disk.
    :param func: Some arbitrary function that performs a complex computation
    :return: A new wrapped function
    """
    @wraps(func)
    def _pickle_store(**kwargs):

        # Check if result is already in cache
        cache_fname = str(pathlib.Path.joinpath(cache_location, func.__name__ + ".pcl"))
        max_bytes = 2 ** 31 - 1
        if os.path.exists(cache_fname):
            try:
                input_size = os.path.getsize(cache_fname)
                bytes_in = bytearray(0)
                with open(cache_fname, "rb") as f:
                    for _ in trange(0, input_size, max_bytes):
                        bytes_in += f.read(max_bytes)
                logging.info("Unpickling cache file..")
                dump = pcl.loads(bytes_in)
                logging.info("Finished unpickling cache file")
            except OSError:
                logging.error("Faulty cache file!")
                sys.exit(1)
        else:
            # Compute
            dump = func(**kwargs)

            # Save
            bytes_out = pcl.dumps(dump, protocol=pcl.HIGHEST_PROTOCOL)
            n_bytes = sys.getsizeof(bytes_out)
            with open(cache_fname, "wb") as f:
                for idx in trange(0, n_bytes, max_bytes):
                    f.write(bytes_out[idx:idx + max_bytes])

        return dump

    return _pickle_store
import logging
import functools
import os


@functools.lru_cache()
def create_logger_dist(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(f"{name}_{dist_rank}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    # color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
    #             colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'{name}_{dist_rank}.log'), mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    return logger
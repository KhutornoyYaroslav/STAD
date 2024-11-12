import os
import sys
import logging


console_logging_level = logging.INFO


def setup_logger(name, distributed_rank, save_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(console_logging_level)

    # do not log results for the non-master process
    if distributed_rank > 0:
        return logger

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if save_dir:
        log_file = os.path.join(save_dir, 'logs.txt')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("Log file: %s" % log_file)

    return logger

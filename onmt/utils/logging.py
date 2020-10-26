# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
from logging.handlers import RotatingFileHandler
logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET, maxBytes=1000, backupCount=10):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = RotatingFileHandler(
            log_file, maxBytes=maxBytes, backupCount=backupCount)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

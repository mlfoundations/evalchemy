import logging
import os
from pathlib import Path

from utils.path_utils import PathUtils


class LogUtils:

    LOGGER_FORMATTER = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    LOGGER_DICT = {}

    def __init__(self):
        pass

    @staticmethod
    def get_or_init_logger(file_name, dir_name=None, level=logging.DEBUG) -> logging.Logger:
        """
        log: time-level-prefix-msg
        :param file_name:
        :param prefix:
        :param dir_name:
        :param level:
        :return:
        """
        if dir_name is None:
            raise ValueError("job id should not be None!")

        log_dir_path = "{}/{}".format(PathUtils.Log_HOME_PATH, dir_name)
        log_file = "{}/log_{}.log".format(log_dir_path, file_name)

        # return the logger if exists
        if log_file in LogUtils.LOGGER_DICT:
            return LogUtils.LOGGER_DICT[log_file]

        # create a logger if not exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler = logging.FileHandler(log_file, mode='a')
        handler.setFormatter(LogUtils.LOGGER_FORMATTER)
        logger = logging.getLogger(file_name)
        logger.setLevel(level)
        logger.addHandler(handler)

        LogUtils.LOGGER_DICT[log_file] = logger
        os.utime(Path(log_dir_path))

        return logger

    @staticmethod
    def get_stat_from_dict(stat_dict, keys=None, is_simple=True):
        if not keys:
            keys = stat_dict.keys()

        msg = []
        for key in keys:
            if is_simple:
                msg.append(stat_dict[key].simple_repr())
            else:
                msg.append(str(stat_dict[key]))

        return "; ".join(msg)


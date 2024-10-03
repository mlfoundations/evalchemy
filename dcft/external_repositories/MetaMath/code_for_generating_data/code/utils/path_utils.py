
import os
import torch


class PathUtils(object):

    def __init__(self):
        pass

    HOME_PATH = ""
    DATA_HOME_PATH = os.path.join(HOME_PATH, "data")
    Log_HOME_PATH = os.path.join(HOME_PATH, "log")
    CONFIG_HOME_PATH = os.path.join(HOME_PATH, "configs")
    FIGS_HOME_PATH = os.path.join(HOME_PATH, "figs")

    @staticmethod
    def set_path():
        PathUtils.DATA_HOME_PATH = os.path.join(PathUtils.HOME_PATH, "data")
        PathUtils.Log_HOME_PATH = os.path.join(PathUtils.HOME_PATH, "log")
        PathUtils.CONFIG_HOME_PATH = os.path.join(PathUtils.HOME_PATH, "configs")
        PathUtils.FIGS_HOME_PATH = os.path.join(PathUtils.HOME_PATH, "figs")

    @staticmethod
    def exists(file_name):
        return os.path.exists(file_name)

    @staticmethod
    def create_dir(file_full_path):
        os.makedirs(os.path.dirname(file_full_path), exist_ok=True)

    @staticmethod
    def get_job_home_path(dir_name):
        _path = '{}/{}/'.format(PathUtils.Log_HOME_PATH, dir_name)
        PathUtils.create_dir(_path)
        return _path

    @staticmethod
    def get_log_home_path():
        return PathUtils.Log_HOME_PATH

    @staticmethod
    def save_ckp(ckp, dir_name, identifier, index_key):
        _path = "{}/{}/ckp-{}-{}".format(PathUtils.Log_HOME_PATH, dir_name, identifier, index_key)
        torch.save(ckp, _path)

    @staticmethod
    def get_file_path(task_name, file_name):
        return "{}/{}/{}".format(PathUtils.Log_HOME_PATH, task_name, file_name)

    @staticmethod
    def get_a_local_file_from_logpath(dir_name, identifier, index_key, file_name):
        return "{}/{}/{}-{}-{}".format(PathUtils.Log_HOME_PATH, dir_name, identifier, index_key, file_name)

    @staticmethod
    def save_ckp_to_path(ckp, _path):
        torch.save(ckp, _path)

    @staticmethod
    def load_ckp(dir_name, identifier, index_key):
        _path = "{}/{}/ckp-{}-{}".format(PathUtils.Log_HOME_PATH, dir_name, identifier, index_key)
        return torch.load(_path, map_location=torch.device('cpu'))

    @staticmethod
    def load_ckp_from_path(_path):
        return torch.load(_path, map_location=torch.device('cpu'))





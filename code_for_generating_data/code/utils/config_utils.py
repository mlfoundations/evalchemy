from utils.path_utils import PathUtils

import yaml
import torch
import os


class ConfigUtils(object):

    def __init__(self):
        pass

    @staticmethod
    def get_device(device_id=0):
        device = torch.device("cpu")
        if torch.cuda.is_available():
            print("GPU is available, using GPU:{}".format(device_id))
            device = torch.device('cuda:{}'.format(device_id))
        else:
            print("GPU is unavailable, using CPU")
        return device

    @staticmethod
    def get_config_dict(config_file_name):
        config_file_full_path = os.path.join(PathUtils.CONFIG_HOME_PATH, config_file_name)
        return yaml.load(open(config_file_full_path, "r"), Loader=yaml.Loader)

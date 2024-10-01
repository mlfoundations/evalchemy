from box import Box
import jsonpickle
import os
import yaml

class BaseConfig(object):
    def __init__(self, to_box=False):
        pass

    def _to_box(self):
        # Convert all dicts to boxes
        for key, val in self.__dict__.items():
            if isinstance(val, dict):
                self.__setattr__(key, Box(val))

    def _from_box(self):
        # Convert all dicts to boxes
        for key, val in self.__dict__.items():
            if isinstance(val, Box):
                self.__setattr__(key, val.to_dict())

    def save(self, save_path=None):
        r''' Save the config into a .json file
        Args:
            save_path (`str`):
                Where to save the created json file, if nothing we use the default from paths.
        '''
        if save_path is None:
            save_path = self.path.self

        # We want to save the dict here, not the whole class
        self._from_box()
        json_string = jsonpickle.encode({k:v for k,v in self.__dict__.items() if k != 'path'})

        with open(save_path, 'w') as f:
            f.write(json_string)
        self._to_box()

    @classmethod
    def load(cls, save_path):
        config = cls(to_box=False)
        # Read the jsonpickle string
        with open(save_path) as f:
            config_dict = jsonpickle.decode(f.read())
        config.merge_config(config_dict)
        config._to_box()
        return config

    def merge_config(self, config_dict):
        r''' Merge a config_dict with the existing config object.
        Args:
            config_dict (`dict`):
                A dictionary which key/values should be added to this class.
        '''
        for key in config_dict.keys():
            if key in self.__dict__ and isinstance(self.__dict__[key], dict):
                self.__dict__[key].update(config_dict[key])
            else:
                self.__dict__[key] = config_dict[key]


class Config(BaseConfig):
    r''' There are probably nicer ways to do this, but I like this one.
    '''
    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        self.load_yaml(yaml_path)

    def reload_yaml(self):
        self.load_yaml(self.yaml_path)

    def load_yaml(self, yaml_path):
        _config = yaml.safe_load(open(yaml_path, 'r'))
        self.to_box = True
        self.base_path = './'
        self.datasets = {}
        self.name = 'opengpt'

        for k,v in _config.items():
            self.__setattr__(k, v)
        # For fun, we will also keept the _config
        self._config = _config

        self.path = {'self': os.path.join(self.base_path, f'config_for_{self.name}.json')}
        if _config.get('static_paths', None):
            self.path.update(_config['static_paths'])

        if self.to_box:
            self._to_box()

            def create_dirs(paths):
                for path in paths:
                    if isinstance(path, str):
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                    elif isinstance(path, dict):
                        create_dirs(path.values())
            create_dirs(self.path.values())
        
        # Create dirs for datasets, this is where all the data from one dataset will go
        for ds in self.datasets:
            os.makedirs(os.path.join(self.base_path, ds['name']), exist_ok=True)
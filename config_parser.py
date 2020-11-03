from os import getcwd
from os.path import join, dirname

from yaml import safe_load as load, YAMLError


class MetaConfigTrain(type):
    @property
    def Data(cls):
        return cls.data

    @property
    def Model(cls):
        return cls.model

    @property
    def Train(cls):
        return cls.train 

    @property
    def Optim(cls):
        return cls.optim


class ConfigTrain(metaclass=MetaConfigTrain):
    path_config_file = join(getcwd(), "config/config.yaml")
    # path_config_file = join(dirname(getcwd()), "config/config.yaml")
    with open(path_config_file, 'r') as stream:
        try:
            config_parser = load(stream)
        except YAMLError as error:
            print(f"Can't load config.yaml Error:{error}")

    data = config_parser["Data"]

    model = config_parser["Model"]

    train = config_parser["Train"]

    optim = config_parser["Optim"]


class MetaConfigTest(type):
    @property
    def Data(cls):
        return cls.data

    @property
    def Model(cls):
        return cls.model

    @property
    def Weight(cls):
        return cls.weight

class ConfigTest(metaclass=MetaConfigTest):
    path_config_file = join(getcwd(), "config/config_test.yaml")
    # path_config_file = join(dirname(getcwd()), "config/config.yaml")
    with open(path_config_file, 'r') as stream:
        try:
            config_parser = load(stream)
        except YAMLError as error:
            print(f"Can't load config.yaml Error:{error}")

    data = config_parser["Data"]
    model = config_parser["Model"]
    weight = config_parser["Weight"]


if __name__ == '__main__':
    print(join(dirname(getcwd()), "config/config.yaml"))



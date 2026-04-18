import yaml

class ConfigParser:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self._parse_config()

    def _parse_config(self):
        with open(self.config_file, "r") as stream:
            return yaml.safe_load(stream)

    def get_config(self):
        return self.config
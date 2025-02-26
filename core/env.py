import os


class EnvParser:
    def __init__(self, file_path):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.file_path = os.path.join(root, file_path)
        self.variables = {}
        self._parse_env_file()

    def _parse_env_file(self):
        try:
            with open(self.file_path, "r") as file:
                for line in file:
                    # Strip whitespace and skip comments or empty lines
                    line = line.strip()
                    if line.startswith("#") or not line:
                        continue

                    # Split on the first '=' to separate key and value
                    if "=" in line:
                        key, value = line.split("=", 1)
                        self.variables[key.strip()] = value.strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{self.file_path}' does not exist.")

    def get(self, key, default=None):
        return self.variables.get(key, default)

    def __getitem__(self, key):
        return self.variables[key]


environ = EnvParser(os.environ.get("ENV"))

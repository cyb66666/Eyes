import json
import os


class Config:
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self._data = json.load(f)



    def __str__(self):
        return str(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __getattr__(self, key):
        if key in self._data:
            value = self._data[key]
            if isinstance(value, dict):
                return Config.from_dict(value)
            return value
        raise AttributeError(f"配置项({key})不存在。")

    @classmethod
    def from_dict(cls, data):
        c = cls.__new__(cls)
        c._data = data
        return c


conf = Config("./config.json")

if __name__ == "__main__":
    print(conf.data_path)

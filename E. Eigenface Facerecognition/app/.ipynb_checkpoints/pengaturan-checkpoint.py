import os
import json

class Pengaturan():
    def get_config(self, Key, Path='', Name='config.json'):
        with open(Path + Name) as json_config:
            json_object = json.load(json_config)

        return json_object[Key]


    def write_config(self, Data, Path='', Name='config.json', append=False):
        mode = 'a+' if append else 'w'
        full_path = Path + Name

        with open(full_path, mode=mode) as json_config:
            json.dump(Data, json.load(json_config) if append else json_config)

        return 'success' 

    def update_config(self, Key, Value, Path='', Name='config.json'):
        with open(Path + Name) as json_config:
            json_object = json.load(json_config)

        json_object[Key] = Value

        with open(Path + Name, mode='w') as json_config:
            json.dump(json_object, json_config)

        return 'success' 
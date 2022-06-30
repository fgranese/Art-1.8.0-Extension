import yaml


def read_file(file_path: str):
    with open(file_path, "r") as yaml_file:
        data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        print("Config read successful")
    return data

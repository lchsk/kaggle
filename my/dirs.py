import os


def get_root_dir(in_competition: bool, file: str):
    if in_competition:
        return os.path.abspath(os.path.join(os.path.dirname(file), '../..'))

    return os.path.abspath(os.path.join(os.path.dirname(file), '.'))


def get_data_dir(competition, file):
    return os.path.join(get_root_dir(in_competition=True, file=file), 'data/', competition)


def join_paths(*paths):
    return os.path.join(*paths)
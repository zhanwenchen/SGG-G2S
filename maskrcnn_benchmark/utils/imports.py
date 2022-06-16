# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from importlib.util import spec_from_file_location, module_from_spec
from sys import modules


# from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
def import_file(module_name, file_path, make_importable=False):
    spec = spec_from_file_location(module_name, file_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        modules[module_name] = module
    return module

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from glob import glob as glob_glob
from os.path import (
    dirname as os_path_dirname,
    abspath as os_path_abspath,
    join as os_path_join,
)
from torch.cuda import is_available as cuda_is_available

try:
    from torch.utils.cpp_extension import load as load_ext
    from torch.utils.cpp_extension import CUDA_HOME
except ImportError:
    raise ImportError("The cpp layer extensions requires PyTorch 0.4 or higher")


def _load_C_extensions():
    this_dir = os_path_dirname(os_path_abspath(__file__))
    this_dir = os_path_dirname(this_dir)
    this_dir = os_path_join(this_dir, "csrc")

    main_file = glob_glob(os_path_join(this_dir, "*.cpp"))
    source_cpu = glob_glob(os_path_join(this_dir, "cpu", "*.cpp"))
    source_cuda = glob_glob(os_path_join(this_dir, "cuda", "*.cu"))

    source = main_file + source_cpu

    extra_cflags = []
    if cuda_is_available() and CUDA_HOME is not None:
        source.extend(source_cuda)
        extra_cflags = ["-DWITH_CUDA"]
    source = [os_path_join(this_dir, s) for s in source]
    extra_include_paths = [this_dir]
    return load_ext(
        "torchvision",
        source,
        extra_cflags=extra_cflags,
        extra_include_paths=extra_include_paths,
    )


_C = _load_C_extensions()

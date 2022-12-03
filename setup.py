import glob

import torch
from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

ext_names = ['custom_batch_norm']

ext_modules = []
for ext_name in ext_names:
    cpp_extension = CppExtension(
        f'pytorch_cpp_extensions.{ext_name}.{ext_name}_cpp',
        glob.glob(f'pytorch_cpp_extensions/{ext_name}/*_cpp.cpp'),
    )
    ext_modules.append(cpp_extension)
    if torch.cuda.is_available():
        cuda_extension = CUDAExtension(
            f'pytorch_cpp_extensions.{ext_name}.{ext_name}_cuda',
            glob.glob(f'pytorch_cpp_extensions/{ext_name}/*_cuda.cu'),
        )
        ext_modules.append(cuda_extension)

setup(
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension,
    },
)

import glob
import os
import sys

from setuptools import find_packages, setup

conda_prefix = os.getenv('CONDA_PREFIX')
assert conda_prefix is not None
sys.path.append(os.path.join(conda_prefix, 'lib/python3.10/site-packages/'))
from torch.utils.cpp_extension import (  # noqa: E402
    BuildExtension,
    CppExtension,
)

ext_names = ['custom_batch_norm']
cpp_ext_modules = [
    CppExtension(
        f'pytorch_cpp_extensions.{ext_name}.{ext_name}_cpp',
        glob.glob(f'pytorch_cpp_extensions/{ext_name}/*_cpp.cpp'),
    ) for ext_name in ext_names
]
ext_modules = cpp_ext_modules

setup(
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension,
    },
    packages=find_packages(include=['pytorch_cpp_extensions*']),
    package_data={
        'pytorch_cpp_extensions': ['*.pyi'],
    },
)

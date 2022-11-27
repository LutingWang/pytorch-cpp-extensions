from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension


setup(
    name='ext',
    ext_modules=[
        CppExtension('custom_batch_norm', ['custom_batch_norm.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
)

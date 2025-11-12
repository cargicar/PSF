import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Environment variables that might be needed if the CUDA path is not configured correctly
# E.g., if you get errors, you might need to set these:
# os.environ['CUDA_HOME'] = '/usr/local/cuda' # or C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x

setup(
    name='emd_ext',  # The name you will use to import the compiled module
    # Define the CUDA extension module
    #  ext_modules=[
    #     CUDAExtension(
    #         name='emd_cuda',
    #         sources=[
    #             'cuda/emd.cpp',
    #             'cuda/emd_kernel.cu',
    #         ],
    ext_modules=[
        CUDAExtension(
            # This is the C++ module name that will be compiled (e.g., 'emd_ext_cuda')
            'emd_ext_cuda', 
            # List of C++ and CUDA source files
            sources=[
                'emd.cpp',
                'emd_kernel.cu'
            ],
            # Compiler arguments for modern C++ standards compatible with PyTorch
            extra_compile_args={
                'cxx': ['-std=c++17'],
                'nvcc': ['-std=c++17']
            }
        ),
    ],
    # Required for custom C++/CUDA compilation
    cmdclass={
        'build_ext': BuildExtension
    },
    version='1.0',
    packages=['emd_ext'], # Dummy package to house the python wrapper
)
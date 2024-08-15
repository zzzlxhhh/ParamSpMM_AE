from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# nvcc_path = subprocess.check_output('which nvcc', shell=True).decode().strip()
# print("environment",nvcc_path)
setup(
    name="ParamSpCONV",
    ext_modules=[
        CUDAExtension(
            name="ParamSpCONV",
            sources=[
                "SpCONV_kernel.cu",
                # 'SpMM.h',
                "SpCONV.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-fopenmp"],
                "CUDA": ["-O3", "-Xcompiler", "-fopenmp"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

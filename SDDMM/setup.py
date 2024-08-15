from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# nvcc_path = subprocess.check_output('which nvcc', shell=True).decode().strip()
# print("environment",nvcc_path)
setup(
    name="ParamSDDMM",
    ext_modules=[
        CUDAExtension(
            name="ParamSDDMM",
            sources=[
                "sddmm_kernel.cu",
                'sddmm_baseline_kernel.cu',
                "sddmm.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-fopenmp"],
                "CUDA": ["-O3", "-Xcompiler", "-fopenmp"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

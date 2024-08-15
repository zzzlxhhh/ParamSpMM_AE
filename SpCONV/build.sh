rm -rf build/
rm -rf dist/
rm -rf *.egg-info
TORCH_CUDA_ARCH_LIST="8.6" python setup.py install  

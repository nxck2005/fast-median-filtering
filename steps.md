# For container:
cd ~/projects/fast-median-filtering/src
podman pull nvidia/cuda:12.5.1-devel-ubuntu22.04
podman run --rm -it --device nvidia.com/gpu=all -v ./_:/src nvidia/cuda:12.5.1-devel-ubuntu22.04 bash
cd /src
nvcc testcuda.cu -o test
./test
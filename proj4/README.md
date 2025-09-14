- NVIDIA Cuda 설치
    https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe

- 가상환경 세팅
    cmd> conda create -n vc python==3.10 -y
    cmd> conda activate vc
    vc> pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
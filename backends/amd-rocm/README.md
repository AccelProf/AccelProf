AMD Development Setup

```shell
# Add ROCm repository
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.3.60304-1_all.deb
sudo dpkg -i amdgpu-install_6.3.60304-1_all.deb
sudo apt update
sudo apt --fix-missing install

# Install ROCm core components (without driver)
sudo amdgpu-install --usecase=rocm --no-dkms

# set env
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=/opt/rocm:$CMAKE_PREFIX_PATH
export HIP_PATH=/opt/rocm/hip
export hip_DIR=/opt/rocm/hip
```
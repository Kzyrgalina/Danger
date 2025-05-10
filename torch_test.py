import numpy as np
import torch

print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")

# Test GPU tensor creation
x = torch.randn(3, 3).cuda()
print("\nGPU Tensor test:")
print(x)
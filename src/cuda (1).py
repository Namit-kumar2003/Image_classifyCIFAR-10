import torch
print("CUDA available:", torch.cuda.is_available())
print("Current Device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
print("GPU Name:", torch.cuda.get_device_name() if torch.cuda.is_available() else "No GPU")

import torch
print(hasattr(torch.backends, 'mps'))
try:
    print(torch.backends.mps.is_available())
except Exception as e:
    print(e)


import torch, sys
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("cudnn version:", torch.backends.cudnn.version())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
x = torch.rand(1024,1024, device=("cuda:0" if torch.cuda.is_available() else "cpu"))
print("tensor device:", x.device)


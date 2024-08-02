import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = ['gradio', 'ultralytics', 'opencv-python', 'torch', 'torchvision', 'pytube', 'mss', 'Pillow']

for package in packages:
    install(package)

print("All dependencies installed successfully!")

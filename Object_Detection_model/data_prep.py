# Install all dependencies
!pip install ultralytics tensorflow opencv-python matplotlib seaborn pandas pillow pyyaml tqdm --quiet

import os, sys, shutil, glob, random, yaml, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import torch

# VUerify GP
import torch
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU      : {torch.cuda.get_device_name(0)}")

import ultralytics
from ultralytics import YOLO
ultralytics.checks()

# Verify GPU and set device variable
device = "0" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

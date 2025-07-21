import numpy as np 
import pandas as pd 
import torch 
import math 
import torch.nn.functional as F
import os

LEVELS = [1000, 950, 850, 700, 600, 500, 450, 400, 300, 250, 200, 150, 100]


def load_static(var="dem"):
    current_path = os.getcwd()
    print(current_path)
    file = os.path.abspath(f"{current_path}/share/china_{var}.npz")
    data = np.load(file)[var]
    return data

# x = load_static()
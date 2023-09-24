import sys
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from torchvision.models import resnet18
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from plotly.subplots import make_subplots
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
import itertools
import matplotlib.pyplot as plt
import faiss
from sklearn.metrics import roc_curve, auc

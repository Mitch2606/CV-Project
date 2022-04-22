import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import torchvision
import torch

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

imageSize = 500;

# Creating a DeepAutoencoder class
class DeepAutoencoder(torch.nn.Module):
	def __init__(self):
		super().__init__()		
		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(imageSize * imageSize, 256),
			torch.nn.ReLU(),
			torch.nn.Linear(256, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 10),
			torch.nn.ReLU(),
			torch.nn.Linear(10, 3),
			
		)
		
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(3, 10),
			torch.nn.ReLU(),
			torch.nn.Linear(10, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 256),
			torch.nn.ReLU(),
			torch.nn.Linear(256, imageSize * imageSize),
			torch.nn.Sigmoid()
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded
	
	def getEncoded(self, x):
		encoded = self.encoder(x)
		return encoded

	def getDecoded(self, encoded_x):
		decoded = self.decoded(encoded_x)
		return decoded
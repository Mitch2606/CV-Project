
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import torchvision
import torch


import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from deepAutoEncoderClass import DeepAutoencoder

# Instantiating the model and hyperparameters
imageSize = 500;
model = DeepAutoencoder()
criterion = torch.nn.MSELoss()
num_epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


from datasetClass import myDataset
dataset = myDataset(csv_file = "SegmentedImageset.csv", root_dir='itemDataset', transform=transforms.ToTensor())

batch_size = 32;
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [417, 105])
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True);
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True);

# Printing 25 random images from the training dataset
random_samples = np.random.randint(
	1, len(train_dataset), (25))

img = train_dataset[0][0][0]
temp = img.reshape(-1, imageSize * imageSize)

a = model.getEncoded(temp)
print(a)


for idx in range(random_samples.shape[0]):
	plt.subplot(5, 5, idx + 1)
	plt.imshow(train_dataset[idx][0][0].numpy(), cmap='gray')
	plt.title(train_dataset[idx][1])
	plt.axis('off')

plt.tight_layout()
plt.show()

# List that will store the training loss
train_loss = []

# Dictionary that will store the
# different images and outputs for
# various epochs
outputs = {}

batch_size = len(train_loader)

# Training loop starts
for epoch in range(num_epochs):
	print("Epoch: " + str(epoch))
	# Initializing variable for storing
	# loss
	running_loss = 0
	
	# Iterating over the training dataset
	for batch in train_loader:
			
		# Loading image(s) and
		# reshaping it into a 1-d vector
		img, _ = batch
		img = img.reshape(-1, imageSize * imageSize)
		
		# Generating output
		out = model(img)
		
		# Calculating loss
		loss = criterion(out, img)
		
		# Updating weights according
		# to the calculated loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		# Incrementing loss
		running_loss += loss.item()
	
	# Averaging out loss over entire batch
	running_loss /= batch_size
	train_loss.append(running_loss)
	
	# Storing useful images and
	# reconstructed outputs for the last batch
	outputs[epoch+1] = {'img': img, 'out': out}

torch.save(model, "newModel.pt")

'''
# Plotting the training loss
plt.plot(range(1,num_epochs+1),train_loss)
plt.xlabel("Number of epochs")
plt.ylabel("Training Loss")
plt.show()

# Plotting is done on a 7x5 subplot
# Plotting the reconstructed images

# Initializing subplot counter
counter = 1

# Plotting reconstructions
# for epochs = [1, 5, 10, 50, 100]
epochs_list = [1, 5, 10, 50, 100]

# Iterating over specified epochs
for val in epochs_list:
	
	# Extracting recorded information
	temp = outputs[val]['out'].detach().numpy()
	title_text = f"Epoch = {val}"
	
	# Plotting first five images of the last batch
	for idx in range(3):
		plt.subplot(7, 5, counter)
		plt.title(title_text)
		plt.imshow(temp[idx].reshape(imageSize, imageSize), cmap= 'gray')
		plt.axis('off')
		
		# Incrementing the subplot counter
		counter+=1

# Plotting original images

# Iterating over first five
# images of the last batch
for idx in range(5):
	
	# Obtaining image from the dictionary
	val = outputs[10]['img']
	
	# Plotting image
	plt.subplot(7,5,counter)
	plt.imshow(val[idx].reshape(imageSize, imageSize),
			cmap = 'gray')
	plt.title("Original Image")
	plt.axis('off')
	
	# Incrementing subplot counter
	counter+=1

plt.tight_layout()
plt.show()

# Dictionary that will store the different
# images and outputs for various epochs
outputs = {}

# Extracting the last batch from the test
# dataset
img, _ = list(test_loader)[-1]

# Reshaping into 1d vector
img = img.reshape(-1, imageSize * imageSize)

# Generating output for the obtained
# batch
out = model(img)

# Storing information in dictionary
outputs['img'] = img
outputs['out'] = out

# Plotting reconstructed images
# Initializing subplot counter
counter = 1
val = outputs['out'].detach().numpy()

# Plotting first 10 images of the batch
for idx in range(10):
	plt.subplot(2, 10, counter)
	plt.title("Reconstructed \n image")
	plt.imshow(val[idx].reshape(imageSize, imageSize), cmap='gray')
	plt.axis('off')

	# Incrementing subplot counter
	counter += 1

# Plotting original images

# Plotting first 10 images
for idx in range(10):
	val = outputs['img']
	plt.subplot(2, 10, counter)
	plt.imshow(val[idx].reshape(imageSize, imageSize), cmap='gray')
	plt.title("Original Image")
	plt.axis('off')

	# Incrementing subplot counter
	counter += 1

plt.tight_layout()
plt.show()
'''
from torch.utils import data
import config
import create_dataloader
from imutils import paths
from torchvision.models import resnet50
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

training_tansform = transforms.Compose([
	transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(90),
	transforms.ToTensor(),
	#transforms.Normalize(mean=config.MEAN, std=config.STD)
])
validation_transform = transforms.Compose([
	transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
	transforms.ToTensor(),
	#transforms.Normalize(mean=config.MEAN, std=config.STD)
])

# create data loaders
(training_ds, training_loader) = create_dataloader.get_dataloader(config.TRAIN,
	transforms=training_tansform,
	batch_size=config.FEATURE_EXTRACTION_BATCH_SIZE)
(val_ds, val_loader) = create_dataloader.get_dataloader(config.VAL,
	transforms=validation_transform,
	batch_size=config.FEATURE_EXTRACTION_BATCH_SIZE, shuffle=False)

classes = ('Fundus', 'Not Fundus')

dataiter = iter(training_loader)
images, labels = dataiter.next()


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.view(-1, 16 * 61 * 61)            
        x = F.relu(self.fc1(x))               
        x = F.relu(self.fc2(x))               
        x = self.fc3(x)                       
        return x

model = ConvNet().to(config.DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=config.LR)

n_total_steps = len(training_loader)

print('t')

for epoch in range(config.EPOCHS):
	print('s')
	for i, (images, labels) in enumerate(training_loader):
		images = images.to(config.DEVICE)
		labels = labels.to(config.DEVICE)
	
        # Forward pass
		outputs = model(images)
		loss = criterion(outputs, labels)

        # Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	
		print(f'Epoch [{epoch+1}/{config.EPOCHS}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)



with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in val_loader:
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        
        # for i in range(config.FEATURE_EXTRACTION_BATCH_SIZE):
        #     label = labels[i]
        #     pred = predicted[i]
        #     if (label == pred):
        #         n_class_correct[label] += 1
        #     n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    # for i in range(10):
    #     acc = 100.0 * n_class_correct[i] / n_class_samples[i]
    #     print(f'Accuracy of {classes[i]}: {acc} %')
# -*- coding: utf-8 -*-
# @Time    : 2021/3/28 19:54
# @Author  : zxf
import numpy as np

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import SimpleModel
from import_libraries_and_generate_data import SimpleDataset

# Generate simulated data
train_size = 6000
test_size = 1000

input_size = 20
hidden_sizes = [50, 50]
output_size = 1
num_classes = 2

X_train = np.random.randn(train_size, input_size).astype(np.float32)
X_test = np.random.randn(test_size, input_size).astype(np.float32)
y_train = np.random.randint(num_classes, size=train_size)
y_test = np.random.randint(num_classes, size=test_size)
print('Shape of X_train:', X_train.shape)
print('Shape of X_train:', X_test.shape)
print('Shape of y_train:', y_train.shape)
print('Shape of y_test:', y_test.shape)

# Create DataLoaders for training and test set, for batch training and evaluation
train_loader = DataLoader(dataset=SimpleDataset(X_train, y_train),
                          batch_size=8, shuffle=True)
test_loader = DataLoader(dataset=SimpleDataset(X_test, y_test),
                         batch_size=8, shuffle=False)

# Set device to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device used:', device)
model_pytorch = SimpleModel(input_size=input_size, hidden_sizes=hidden_sizes,
                            output_size=output_size)
model_pytorch = model_pytorch.to(device)

# Set loss and optimizer
# Set binary cross entropy loss since 2 classes only
criterion = nn.BCELoss()
optimizer = optim.Adam(model_pytorch.parameters(), lr=1e-3)

num_epochs = 20

# Train model
time_start = time.time()

for epoch in range(num_epochs):
    model_pytorch.train()

    train_loss_total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.float().to(device)
        optimizer.zero_grad()
        output = model_pytorch(data)
        train_loss = criterion(output, target)
        train_loss.backward()
        optimizer.step()
        train_loss_total += train_loss.item() * data.size(0)

    print('Epoch {} completed. Train loss is {:.3f}'.format(epoch + 1, train_loss_total / train_size))
print('Time taken to completed {} epochs: {:.2f} minutes'.format(num_epochs, (time.time() - time_start) / 60))

# Evaluate model
model_pytorch.eval()

test_loss_total = 0
total_num_corrects = 0
threshold = 0.5
time_start = time.time()

for data, target in test_loader:
    data, target = data.to(device), target.float().to(device)
    optimizer.zero_grad()
    output = model_pytorch(data)
    train_loss = criterion(output, target)
    train_loss.backward()
    optimizer.step()
    train_loss_total += train_loss.item() * data.size(0)

    pred = (output >= threshold).view_as(target)  # to make pred have same shape as target
    num_correct = torch.sum(pred == target.byte()).item()
    total_num_corrects += num_correct

print('Evaluation completed. Test loss is {:.3f}'.format(test_loss_total / test_size))
print('Test accuracy is {:.3f}'.format(total_num_corrects / test_size))
print('Time taken to complete evaluation: {:.2f} minutes'.format((time.time() - time_start) / 60))

if not os.path.exists('./models/'):
    os.mkdir('./models/')

torch.save(model_pytorch.state_dict(), './models/model_simple.pt')
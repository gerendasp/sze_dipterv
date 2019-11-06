import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models, transforms
from barbar import Bar
from sklearn.metrics import accuracy_score


# Download and preprocess images
transforms = {'train':transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()]),
        'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]) }

# Create datasets from images
datatsets = {'train':torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms['train']),
             'val':torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms['val'])}

dataset_sizes = {'train':len(datatsets['train']),'val':len(datatsets['val'])}


# Data loaders
dataloaders = {
'train':torch.utils.data.DataLoader(datatsets['train'], batch_size=64, num_workers=0),
'val':torch.utils.data.DataLoader(datatsets['val'], batch_size=64, num_workers=0)
}

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10
resnet = models.densenet161(num_classes = num_classes)


# Parameters for training
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training
model = resnet

def train_step(x, t):
        model.train()
        preds = model(x)
        loss = criterion(preds, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, preds

epochs = 30

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch+1))
    train_loss = 0.
    train_acc = 0.

    for idx, (x, t) in enumerate(Bar(dataloaders['train'])):
        x, t = x.to(device), t.to(device)
        loss, preds = train_step(x, t)
        train_loss += loss.item()
        train_acc += \
            accuracy_score(t.tolist(),
                           preds.argmax(dim=-1).tolist())

    train_loss /= len(dataloaders['train'])
    train_acc /= len(dataloaders['train'])

    print('loss: {:.3}, acc: {:.3f}'.format(train_loss, train_acc))

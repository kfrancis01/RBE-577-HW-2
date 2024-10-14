import json
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import time
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch import Tensor
from typing import Any, Callable, List, Optional, Type, Union
from torchvision.models import resnet152
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Define your Params class here
class Params:
    def __init__(self):
        self.batch_size = 16
        # self.batch_size = 32
        self.name = "resnet_152_sgd1"
        self.workers = 4
        self.lr = 0.001
        self.momentum = 0.9
        self.weight_decay = 1e-5
        self.lr_step_size = 30
        self.lr_gamma = 0.1

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

# Custom dataset for test images without subfolders
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(".jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0

# Function to save image and its classification
def show_classification(image, prediction, class_names, index):
    plt.imshow(image.permute(1, 2, 0))  # Convert from Tensor to image format (HWC)
    plt.title(f"Predicted: {class_names[prediction]}")
    plt.axis('off')
    
    # Ensure directory exists
    output_dir = "classified_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the image
    plt.savefig(os.path.join(output_dir, f"classification_image_{index}.png"))
    plt.close()  # Close the figure to avoid display


# Function to classify the first 10 images and save them
def classify_and_save_images(test_loader, model, class_names):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for idx, (inputs, _) in enumerate(test_loader):
            if idx >= 10:
                break  # Stop after 10 images
            inputs = inputs.to(device)

            # Get model predictions
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Loop through the batch (in case batch size > 1)
            for i in range(inputs.size(0)):
                if idx * params.batch_size + i >= 10:
                    break  # Ensure we only process 10 images
                
                # Save the image with its classification (use class_names)
                show_classification(inputs[i].cpu(), predicted[i].cpu().item(), class_names, idx * params.batch_size + i)

# Training function
def train(dataloader, model, loss_fn, optimizer, epoch, writer):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(X)
        loss = loss_fn(outputs, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    epoch_loss = running_loss / size
    epoch_accuracy = correct / total

    writer.add_scalar('Loss/Train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/Train', epoch_accuracy, epoch)

    print(f"Epoch {epoch} Training Accuracy: {epoch_accuracy:.4f}, Loss: {epoch_loss:.4f}")

# Validation function
def validate(loader, model, loss_fn, epoch, writer):
    size = len(loader.dataset)
    num_batches = len(loader)
    model.eval()
    val_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    val_loss /= num_batches
    val_accuracy = correct / size

    writer.add_scalar('Loss/Validation', val_loss, epoch)
    writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

    print(f"Epoch {epoch} Validation Accuracy: {val_accuracy:.4f}, Loss: {val_loss:.4f}")
    return val_accuracy

# Test function to display classification for 10 images
def test_and_show(loader, model, label_map):
    model.eval()
    images_shown = 0

    with torch.no_grad():
        for X, _ in loader:
            if images_shown >= 10:
                break
            X = X.to(device)
            predictions = model(X).argmax(1)

            # Loop through each image in the batch and show the classification
            for i in range(X.shape[0]):
                if images_shown >= 10:
                    break
                show_classification(X[i].cpu(), predictions[i].cpu().item(), label_map)
                images_shown += 1

if __name__ == '__main__':
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    params = Params()

    # Load datasets and transformations
    dir = "/Users/keafrancis/Documents/WPI/RBE 577/HW 2/Data/"
    training_folder_name = dir + 'train'
    val_folder_name = dir + 'val'
    test_folder_name = dir + 'test'

    train_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=training_folder_name, transform=train_transformation)
    val_dataset = torchvision.datasets.ImageFolder(root=val_folder_name, transform=val_test_transformation)
    test_dataset = CustomImageDataset(image_dir=test_folder_name, transform=test_transform)
    
    # Extract class names from train dataset
    class_names = train_dataset.classes

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, num_workers=params.workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params.batch_size, num_workers=params.workers, pin_memory=True, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, num_workers=params.workers, pin_memory=True, shuffle=False)  # Batch size = 10 for test

    # Load ResNet152 pretrained model
    model = resnet152(pretrained=True)

    # Freeze ResNet152 backbone
    for param in model.parameters():
        param.requires_grad = False

    # Modify the final layer (fully connected layer) for 10 classes
    num_features = model.fc.in_features
    num_classes = 10  # Set to your custom dataset's number of classes
    model.fc = nn.Sequential(
        nn.Linear(num_features, num_classes),
        nn.Dropout(0.5)
    )

    model = model.to(device)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.lr_step_size, gamma=params.lr_gamma)

    writer = SummaryWriter('runs/' + params.name)

    # Training loop
    label_map = {i: str(i) for i in range(10)}  # Custom label map (update with actual class names if available)
    for epoch in range(1, 50):
        train(train_loader, model, loss_fn, optimizer, epoch, writer)
        validate(val_loader, model, loss_fn, epoch, writer)

        # Adjust learning rate
        lr_scheduler.step()

        # Save model checkpoint
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, os.path.join("checkpoints", f"checkpoint_epoch_{epoch}.pth"))

    # Show classification of 10 images from the test set
    #test_and_show(test_loader, model, label_map)
    
    # Perform classification on 10 test images and save results
    classify_and_save_images(test_loader, model, class_names)
    print(f"Classes: {train_dataset.classes}")

    writer.close()



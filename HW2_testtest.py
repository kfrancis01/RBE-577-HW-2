#!/usr/bin/python3

import os

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import torch.nn.functional as F

from torchvision import datasets, transforms
import matplotlib.pyplot as plt

#import TensorBoard SummaryWriter
#from torch.utils.tensorboard import SummaryWriter

from torchvision.io import read_image
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from PIL import Image

# Set random seed
random_seed = 123
torch.manual_seed(random_seed)

# Hyperparameters
learning_rate = 0.01
num_epochs = 10 
batch_size = 32

# directory for test images
dir = "/Users/keafrancis/Documents/WPI/RBE 577/HW 2/Data/"

# Define image transformations for training, validation, and test sets
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# mnist_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])

# Since the test folder doesn't have subfolders a custom dataset must be created
# https://discuss.pytorch.org/t/using-imagefolder-without-subfolders-labels/67439
# Define a custom dataset class for your test images
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(".jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Return the image and a dummy label (e.g., 0)
        return image, 0

# Load datasets for training, validation, and test sets
# test_dataset = datasets.ImageFolder(os.path.join(dir, 'test'), transform=val_test_transform)
# test_dataset = datasets.ImageFolder(root="/Users/keafrancis/Documents/WPI/RBE 577/HW 2/Data/test", transform=val_test_transform)
test_dir = dir + '/test'
test_dataset = CustomImageDataset(image_dir=test_dir, transform=test_transform)
# train_dataset = datasets.ImageFolder(os.path.join(dir, 'train/bus/0aa8c6bece0ab6ceda43e920634e4fa2.jpg'), transform=train_transform)
train_dataset = datasets.ImageFolder(os.path.join(dir, 'train'), transform=train_transform)
# val_dataset = datasets.ImageFolder(os.path.join(dir, 'val/bus/1b9471eefb6f3951f127beb69ef5a584.jpg'), transform=val_test_transform)
val_dataset = datasets.ImageFolder(os.path.join(dir, 'val'), transform=val_transform)
# test_dataset = datasets.ImageFolder(os.path.join(dir, 'test/0a5067c35b854ce7213e433d12ea500d.jpg'), transform=val_test_transform)

# can not use MNIST since Images are pretrained
# train_dataset = datasets.MNIST(root='data', 
#                                train=True, 
#                                transform=transforms.ToTensor(),
#                                download=True)
# train_dataset = datasets.ImageFolder(os.path.join(dir, 'train'), transform=train_transform)
# # test_dataset = datasets.MNIST(root='data', 
# #                               train=False, 
# #                               transform=transforms.ToTensor())
# test_dir = dir + '/test'
# test_dataset = CustomImageDataset(image_dir=test_dir, transform=test_transform)
# # val_dataset = datasets.MNIST(root='data', 
# #                               train=False, 
# #                               transform=transforms.ToTensor())
# val_dataset = datasets.ImageFolder(os.path.join(dir, 'val'), transform=val_transform)

train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)

val_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)

# Checking the dataset
# for images, labels in train_loader:  
#     print('Image batch dimensions:', images.shape) #NCHW
#     print('Image label dimensions:', labels.shape)
#     break

# # Create data loaders for training, validation, and test sets
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model
# weights = ResNet50_Weights.DEFAULT
# model = resnet50(weights=weights)
# model = model.to(device)

weights = ResNet101_Weights.DEFAULT
model = resnet101(weights=weights)
model = model.to(device)

# Freeze the model parameters
# Replace the final fully connected layer to match the number of classes in your dataset
num_features = model.fc.in_features
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
#criterion = torch.nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Set up TensorBoard SummaryWriter
# log_dir = "runs/experiment_1"  # Change this to your desired log directory
# writer = SummaryWriter(log_dir=log_dir)
#writer = SummaryWriter()

# Helper function to calculate accuracy
def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability

            # Update total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total  # Return accuracy as a ratio

# Training and validation loop
for epoch in range(num_epochs):
    # Training Phase
    model.train()  # Set model to training mode
    running_train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        train_loss = criterion(outputs, labels)

        # Backward pass and optimization
        train_loss.backward()
        optimizer.step()

        # Update running training loss
        running_train_loss += train_loss.item() * inputs.size(0)

    # Calculate average training loss and accuracy for the epoch
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_accuracy = calculate_accuracy(train_loader, model)

    # Validation Phase
    model.eval()  # Set model to evaluation mode
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)

            # Update running validation loss
            running_val_loss += val_loss.item() * inputs.size(0)

    # Calculate average validation loss and accuracy for the epoch
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    val_accuracy = calculate_accuracy(val_loader, model)

    # Log the losses and accuracies to TensorBoard
    # writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
    # writer.add_scalar('Loss/Validation', epoch_val_loss, epoch)
    # writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
    # writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

    # Print the losses and accuracies for monitoring
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}, "
          f"Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Test Phase
test_accuracy = calculate_accuracy(test_loader, model)
# writer.add_scalar('Accuracy/Test', test_accuracy, num_epochs)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Close the TensorBoard writer
#writer.close()

# HW Part 4: 10 Ex of classified images from teh test dataset

# Get a list of all image files in the directory
# image_files = [f for f in os.listdir(dir + '/test') if f.endswith((".jpg"))]

# # Randomly select 10 images from the directory
# random_images = nn.random.sample(image_files, min(10, len(image_files)))  # Select 10 or fewer if not enough images

# Loop through the selected random images and classify them
# for image_file in random_images:
#     # Load the image and apply transformations
#     image_path = os.path.join(dir + '/test', image_file)
#     image = Image.open(image_path).convert("RGB")  # Convert to RGB to ensure 3 channels
#     input_tensor = test_transform(image).unsqueeze(0).to(device)  # Add batch dimension

#     # Run the image through the model and get predictions
#     with torch.no_grad():
#         output = model(input_tensor)
    
#     # Get the predicted class index and score
#     _, predicted_idx = torch.max(output, 1)
#     predicted_class = class_names[predicted_idx.item()]
#     predicted_score = torch.nn.functional.softmax(output, dim=1)[0][predicted_idx].item()

#     # Display the image and classification
#     plt.imshow(image)
#     plt.title(f"Predicted: {predicted_class} ({predicted_score:.2f})")
#     plt.axis('off')
#     plt.show()

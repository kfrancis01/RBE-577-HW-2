#!/usr/bin/python3

import time
import os
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import matplotlib
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

test = []
train = []
val = []

def import_images()
    for folder in ['train', 'test', 'val']:
        for images in 
dir = /Users/keafrancis/Documents/WPI/RBE 577/HW 2/Data/

test_img = import_images('test')
train_img = import_images('train') #read_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")
val_img = import_images('val')

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Using weights from a specified source:

# Using pretrained weights:
resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet50(weights="IMAGENET1K_V1")
resnet50(pretrained=True)  # deprecated
resnet50(True)  # deprecated

# Using no weights:
resnet50(weights=None)
resnet50()
resnet50(pretrained=False)  # deprecated
resnet50(False)  # deprecated
    
# Initialize the Weight Transforms
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

for param in model.parameters():
    param.requires_grad = False

# Apply it to the input image
img_transformed = preprocess(img)

# Initialize model
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")

# Set model to eval mode
model.eval()

# List available models
all_models = list_models()
classification_models = list_models(module=torchvision.models)

# Initialize models
m1 = get_model("mobilenet_v3_large", weights=None)
m2 = get_model("quantized_mobilenet_v3_large", weights="DEFAULT")

# Fetch weights
weights = get_weight("MobileNet_V3_Large_QuantizedWeights.DEFAULT")
assert weights == MobileNet_V3_Large_QuantizedWeights.DEFAULT

weights_enum = get_model_weights("quantized_mobilenet_v3_large")
assert weights_enum == MobileNet_V3_Large_QuantizedWeights

weights_enum2 = get_model_weights(torchvision.models.quantization.mobilenet_v3_large)
assert weights_enum == weights_enum2


# Hyperparameters
random_seed = 123
learning_rate = 0.1
num_epochs = 20
batch_size = 256


num_features = 784
num_classes = 10

train_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(),
    transforms.RandomHorizontalFlip(),
    train=True,
    # Normalize pixel values
    transforms.Normalize((0.1307, ), (0.3081, )) //needs to match pretrained model
])

test_transformation = transforms.Compose([
    transforms.ToTensor(),
    train=False,
    transforms.Normalize((0.1307,), (0.3081,)) //needs to match pretrained model
])



# train_dataset = datasets.MNIST(root='data', 
#                                train=True, 
#                                transform=transforms.ToTensor(),
#                                download=True)

# test_dataset = datasets.MNIST(root='data', 
#                               train=False, 
#                               transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)


# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape) #NCHW
    print('Image label dimensions:', labels.shape)
    break

class SoftmaxRegression(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)
        
        
    def forward(self, x):
        logits = self.linear(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

model = SoftmaxRegression(num_features=num_features,
                          num_classes=num_classes)

model.to(device)


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

torch.manual_seed(random_seed)


def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    
    for features, targets in data_loader:
        features = features.view(-1, 28*28).to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
        
    return correct_pred.float() / num_examples * 100
    

start_time = time.time()
epoch_costs = []
for epoch in range(num_epochs):
    avg_cost = 0.
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.view(-1, 28*28).to(device)
        targets = targets.to(device)
            
        logits, probas = model(features)
        
        # the PyTorch implementation ofCrossEntropyLoss works with logits, not probabilities
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        cost.backward()
        avg_cost += cost
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_dataset)//batch_size, cost))
            
    with torch.set_grad_enabled(False):
        avg_cost = avg_cost/len(train_dataset)
        epoch_costs.append(avg_cost)
        print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
              epoch+1, num_epochs, 
              compute_accuracy(model, train_loader)))
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))


epoch_costs = [t.cpu().item() for t in epoch_costs]
plt.plot(epoch_costs)
plt.ylabel('Avg Cross Entropy Loss\n(approximated by averaging over minibatches)')
plt.xlabel('Epoch')
plt.show()

print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))

for features, targets in test_loader:
    print(features.shape)
    print(targets.shape)
    print(targets)
    break
    
fig, ax = plt.subplots(1, 4)
for i in range(4):
    ax[i].imshow(features[i].view(28, 28), cmap=matplotlib.cm.binary)

plt.show()



_, predictions = model.forward(features[:4].view(-1, 28*28).to(device))
predictions = torch.argmax(predictions, dim=1)
print('Predicted labels', predictions)
# RBE-577-HW-2

## Prerequisites
* Python (Version 3.9.13)
* TensorBoard (Version 2.12.0)
* TensorFlow (Version 2.12.0)
* PyTorch (Version 2.2.2)
* Keras (Version 2.12.0)
* macOS ARM64 Compatibility: Ensure you are using the correct ARM64 versions of TensorFlow and TensorBoard, especially if you are using macOS with M1/M2 chips.

## Getting Started
1. (Optional) Open VS Code or your preferred IDE
2. Set Up a Conda Environment:

Create a dedicated conda environment to avoid package conflicts, especially for macOS ARM64:

```
conda create -n arm_tensorboard_env python=3.9
conda activate arm_tensorboard_env
pip install tensorflow-macos==2.12.0 tensorboard==2.12.0 keras==2.12.0 torch==2.2.2 torchvision==0.15.0
```
3. Install necessary Python packages:
```
pip install tensorflow==2.12.0 tensorboard==2.12.0 keras==2.12.0 torch==2.2.2 torchvision==0.15.0
```
4. Verify Installation: After installation, ensure there are no broken dependencies by running
```
pip check
```
5. Running TensorBoard: Start TensorBoard by pointing to the directory where logs are stored
```
cd /path/to/your/project
tensorboard --logdir=runs --port 6020
```
Use the appropriate logdir path based on your project setup. Ensure that VPN or firewall configurations do not block the connection to localhost:6020.
6. View TensorBoard by pasting http://localhost:6020/ to the url


## Data Set Collection
**Kaggle Vehicle Image Data** \
https://www.kaggle.com/datasets/marquis03/vehicle-classification/data

**Pretrained Resnet** \
The pretrained ResNet model (trained on the Imagenet dataset) is used as the base.
https://pytorch.org/vision/stable/models.html

## Methodology and Process
1. **Model Selection**: 
I fine-tuned a pretrained ResNet152 model after comparing it with the results from ResNet50 and ResNet101. The model is loaded with pretrained weights from ImageNet, and the final fully connected layer (the head) is removed and replaced with a custom classifier specific to the vehicle classification dataset.
2. **Remove ResNet Head and Implement Classification Head**:
The ResNet head is removed and replaced with a new fully connected layer that matches the number of output classes. Additionally, dropout is added for regularization to help prevent overfitting:
```
model.fc = nn.Sequential(
    nn.Linear(num_features, num_classes),
    nn.Dropout(0.5)
)
```
3. **Training**
Pretrained layers were frozen to reduce training time and focus on the custom classification head:
```
for param in model.parameters():
    param.requires_grad = False
```
4. **Data Regularization and Augmentation** You need to apply proper regularization and data augmentation techniques that were
discussed in the lectures to avoid overfitting and underfitting.

Input images are normalized with the same mean and standard deviation values used during ImageNet training. 
```transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])```

**Data Regularization**
* **Dropout** is applied in the final classification layer to prevent overfitting.
* **Weight decay** is used to reduce model complexity and overfitting
```weight_decay=1e-4```
* **Momentum** is incorporated in the optimizer to speed up convergence
* **Device** is used inorder to speed up the computation
```device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")``` 
* **Pixel Scaling**:
  
**Data Augmentation**: The dataset was augmentated with transformations like horizontal flipping and resizing to help the model generalize better.
* Use various techniques such as horizontal flips, resizing, and affine transformations to augment the training data.
* Ensure images are normalized using the same statistics (mean and standard deviation) as used during the training of the pretrained ResNet.

## Hyperparameters
* Learning Rate: 0.001
* Momentum: 0.9
* Weight Decay: 1e-4
* Batch Size: 16
* Epochs: 50
* Device: cuda if available, otherwise cpu
  
## Image Classification Examples
**Image I**
This was an impressive prediction since as a human it is very difficult to descern what the image class should be
![alt text](image-1.png)

**Image II**
![alt text](image-2.png)

**Image III**
![alt text](image-3.png)

**Image IV**
![alt text](image-7.png)

**Image V**
![alt text](image-5.png)

**Image VI**
![alt text](image-6.png)

**Image VII**
![alt text](image-8.png)

**Image VIII**
This is an intriguing misclassification during model training. Although the image clearly shows a 'Taxi' sign, the model might have confused it with a sedan, highlighting the challenge of distinguishing between similar vehicle types.
![alt text](image-9.png)

**Image IX**
![alt text](image-10.png)

**Image X**
![alt text](image-11.png)

## Lessons Learned 
1. **Data Loading:** 
* **Custom Dataset Handling**: The MNIST dataset was incompatible with the pretrained ResNet models, which required a different input format. Additionally, when dealing with datasets that don’t follow the typical ImageFolder structure (i.e., directories without subfolders), a custom dataset loader becomes essential. This was particularly important for the test directory, which lacked subdirectories. I found helpful information on implementing a custom dataset loader from the following forum: [PyTorch Discussion](https://discuss.pytorch.org/t/using-imagefolder-without-subfolders-labels/67439).

2. **TensorBoard:** 
* Setting up TensorBoard on macOS ARM64 (Apple Silicon) poses several challenges due to compatibility issues and the need for specific versions of the software.
   TensorBoard and Keras must both be version 2.12.0.
   ```pip install tensorflow==2.12.0```
   ```pip install tensorboard==2.12.0```
   ```pip check``` to make sure there aren't any broken requirements
* VPN can interrupt TensorBoard connecting to Visual Studio Code which I eventually abandoned for the conda environment running through the mac terminal.
* Using the conda environment
I had to ensure that TensorFlow and TensorBoard are installed and running within a properly configured conda environment. For macOS ARM64, creating a new conda environment specifically for TensorFlow and TensorBoard can help mitigate package conflicts:
```
conda create -n arm_tensorboard_env python=3.9
conda activate arm_tensorboard_env
pip install tensorflow-macos==2.12.0 tensorboard==2.12.0
```

3. **Epoch to optimize training** 
Initially, I limited the number of epochs to 10 due to the large dataset size. Increasing the number of epochs revealed that the model wasn't reaching its full potential in terms of performance when limited to 10 epochs. For deeper networks like ResNet152, more epochs were necessary to achieve the best performance.

4. **Model Comparison (ResNet50 vs ResNet101 vs ResNet152)**: 

**resnet50 vs resnet101**
resnet50 outperformed resnet101 achieving better validation accuracy. The overfitting in resnet101 originally made me believe that it was overly complex for the training required.

**resnet50**: Faster with a higher validation accuracy
```
Epoch 1/10, Train Loss: 1.4901, Validation Loss: 0.4182, Train Accuracy: 0.8014, Validation Accuracy: 0.8700
Epoch 2/10, Train Loss: 0.4840, Validation Loss: 0.2859, Train Accuracy: 0.8929, Validation Accuracy: 0.9050
Epoch 3/10, Train Loss: 0.3475, Validation Loss: 0.2800, Train Accuracy: 0.9036, Validation Accuracy: 0.8900
Epoch 4/10, Train Loss: 0.2913, Validation Loss: 0.2479, Train Accuracy: 0.9000, Validation Accuracy: 0.9300
Epoch 5/10, Train Loss: 0.2308, Validation Loss: 0.2693, Train Accuracy: 0.9329, Validation Accuracy: 0.9150
Epoch 6/10, Train Loss: 0.1961, Validation Loss: 0.1537, Train Accuracy: 0.9479, Validation Accuracy: 0.9350
Epoch 7/10, Train Loss: 0.1864, Validation Loss: 0.1661, Train Accuracy: 0.9593, Validation Accuracy: 0.9500
Epoch 8/10, Train Loss: 0.2362, Validation Loss: 0.1630, Train Accuracy: 0.9443, Validation Accuracy: 0.9600
Epoch 9/10, Train Loss: 0.1898, Validation Loss: 0.1246, Train Accuracy: 0.9586, Validation Accuracy: 0.9600
Epoch 10/10, Train Loss: 0.1502, Validation Loss: 0.1097, Train Accuracy: 0.9664, Validation Accuracy: 0.9600
```
**resnet101**: 
```
Epoch 1/10, Train Loss: 1.2955, Validation Loss: 7.1852, Train Accuracy: 0.8371, Validation Accuracy: 0.0750
Epoch 2/10, Train Loss: 0.4082, Validation Loss: 7.5920, Train Accuracy: 0.9079, Validation Accuracy: 0.1050
Epoch 3/10, Train Loss: 0.3128, Validation Loss: 8.3678, Train Accuracy: 0.9379, Validation Accuracy: 0.0950
Epoch 4/10, Train Loss: 0.2193, Validation Loss: 8.1688, Train Accuracy: 0.9514, Validation Accuracy: 0.1150
Epoch 5/10, Train Loss: 0.1954, Validation Loss: 8.3587, Train Accuracy: 0.9514, Validation Accuracy: 0.1050
Epoch 6/10, Train Loss: 0.1440, Validation Loss: 9.9667, Train Accuracy: 0.9721, Validation Accuracy: 0.0950
Epoch 7/10, Train Loss: 0.1445, Validation Loss: 10.2330, Train Accuracy: 0.9636, Validation Accuracy: 0.0950
Epoch 8/10, Train Loss: 0.1310, Validation Loss: 9.5043, Train Accuracy: 0.9686, Validation Accuracy: 0.1000
Epoch 9/10, Train Loss: 0.1079, Validation Loss: 10.1769, Train Accuracy: 0.9693, Validation Accuracy: 0.1000
Epoch 10/10, Train Loss: 0.1214, Validation Loss: 10.0052, Train Accuracy: 0.9821, Validation Accuracy: 0.1000
Test Accuracy: 0.1000
```
**resnet50 vs resnet152**:
Overall resnet152 outperformed resnet50 when it came to accuracy and loss for train, validation, and test. The larger number of neural network layers allows for more depth and complexity when training. This outcome showed that the resnet complexity is not the only thing that should be considered when deciding on the model. How the model performs over time is important when determining the best fit.
![alt text](image.png)

5. **Imporving Test Accuracy (Hyperparameters and Regularization)** 
* Learning Rate Adjustments: Lowering the learning rate from 0.1 to 0.01 showed a significant improvement in test accuracy. Further reductions in the learning rate had diminishing returns, suggesting the optimal value lies between 0.01 and 0.001. 
* Batch Size Effects: Increasing the batch size to 32 (from 16) helped stabilize the test accuracy by reducing the variance in updates, resulting in smoother learning curves.
  
6. **Freezing Layers**
Freezing the pretrained ResNet layers during the initial stages of training proved to be highly beneficial. By locking the weights of the earlier layers, I was able to significantly reduce training time and prevent the model from overfitting to the dataset. 

7. **ARM64 vs x86_64** 
Working with ARM64 on macOS:

    Compatibility issues were resolved by ensuring proper installation of the ARM64-specific versions of TensorFlow and TensorBoard. This setup was critical to avoiding performance degradation or compatibility warnings (e.g., SSE4.2 warnings from Intel-based versions).

## Summary

![alt text](image-12.png)

**Overview**
This project focused on finetuning and comparing pretrained ResNet models (ResNet50, ResNet101, ResNet152) for vehicle classification using a custom dataset from Kaggle. Throughout the process, additional learnings included handling custom datasets, resolving macOS ARM64 compatibility issues, and leveraging TensorBoard for real-time performance tracking.

ResNet152 outperformed the other models due to its deeper architecture, allowing for better feature extraction and generalization. It achieved the highest validation accuracy and the lowest loss.

ResNet50 provided faster training but lacked the complexity needed for the dataset, making it a good choice for smaller datasets or limited computational power.

ResNet101 exhibited overfitting, showing that deeper models aren't always better if not properly regularized or if the dataset size doesn't justify the complexity.

From the final tensborboard graphs the training and validation metrics indicate that the model is performing well in terms of accuracy and loss. Validation accuracy stabilizes at around 94%, and both training and validation loss graphs demonstrate solid learning progress. This is also explified by the 90% classification accuracy for the ten images provided above. 

**Lessons**

Fine-tuning deeper models like ResNet152 can be computationally expensive but yields better accuracy.

Careful hyperparameter tuning, such as adjusting the learning rate and batch size, significantly impacts performance.

Regularization techniques like dropout and weight decay, alongside data augmentation, are crucial to prevent overfitting in complex models.

I was impressed that, even with just a few epochs, the finely tuned model demonstrated strong performance, largely due to the benefits of using pretrained models. However, despite the high overall accuracy, it was interesting to see the model misclassify a taxi, even with a clear 'Taxi' sign. It labeled the vehicle as a family sedan, likely because taxis often share the same structural features and fall under that classification.

This project also highlighted the importance of proper environment setup, especially for ARM64 architecture, and the power of tools like TensorBoard for real-time monitoring and analysis.

**Next Steps**

The test metrics continued to fluctuate which could be caused by noise from overfitting. Further steps such as tuning the batch size, learning rate, or applying better regularization techniques could help address this issue.

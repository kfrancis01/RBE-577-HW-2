# RBE-577-HW-2

## Prerequisites
Python (Version 3.
Tensorboard (Version )
TensorFlow (Version )

## Getting Started
1. Open
2. 


## Data Set Collection
**Kaggle Vehicle Image Data** \
https://www.kaggle.com/datasets/marquis03/vehicle-classification/data

**Pretrained Resnet** \
https://pytorch.org/vision/stable/models.html

## Methodology

## Process
**1.** You need to remove the resnet head and implement a new classification head for this
application. You can either freeze the rest of the resnet weights/biases and only train the
head OR you can finetune all the weights in the resent star ting with the pretrained weights
(obviously the latter requires more GPU).
2. You need to make sure you normalize your images the same way the pretrained
weights are obtained (as discussed in the lecture)
3.  You need to apply proper regularization and data augmentation techniques that were
discussed in the lectures to avoid overfitting and underfitting.
4.  You can use any version of resent such as resnet18, resnet34, resnet50, resnet101,
or resnet152.

## Hyperparameters

## Image Examples of Successful Classification

## Lessons Learnt 

## Summary

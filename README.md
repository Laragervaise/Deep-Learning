# Deep-Learning-Projects

### Required librairies
```
argparse
sys
torch
torchvision
random
math
mathplotlib
```

### How to run it
```
$ python3 test.py 
```
This will print the usage and you'll be able to select and test the model you want.

## Project 1: Classification, weight sharing, auxiliary losses
In this first project, we tested two main  architectures of neural networks: Fully connected neural networks (FNN) and Convolutional neural networks (CNN). Both of the networks  are designed to compare two digits appearing in a two-channeled image taken from the MNIST dataset. The goal of this project is to show in particular the impact of weight sharing (WS) and the use of an auxiliary loss (AUX) to help the training of the main objective. The implementation is in Python, and utilizes the Pytorch library only.

## Project 2: Mini deep-learning framework
In this second project, we designed a mini deep-learning framework, using only pytorch’s tensor operations and the standard math library. In particular, no neural-network modules or autograd was used.
The framework allows to build networks combining the following:
- Fully connected layers,
- ReLU, LeakyReLU, Tanh, Sigmoid and Softmax activation functions,
- Regularization methods such as batch normalization or the use of weight decay,
- Exponential, time-based and step-decay learning rate schedules,
- MSE or Cross entropy loss functions (the latter including the Softmax activation function),
- SGD, Adagrad, RMSProp or Momentum optimizers,
- Random, Zero, He or Xavie initializations.

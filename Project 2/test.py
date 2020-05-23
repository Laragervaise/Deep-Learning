from utils import load_data, model_selector, train, compute_acc
from torch import set_grad_enabled
import argparse
import sys
import torch

# Turn off autograd
set_grad_enabled(False)

# General constants
eps = 1e-5
momentum = 0.9

# Model ids and names
mids = [0,1,2,3,4,5,6,7,8]
model_names = ['ReLU', 'ReLU & Sigmoid', 'ReLU & Tanh', 'ReLU & Softmax',\
               'Leaky ReLU', 'Leaky ReLU & Sigmoid', 'Leaky ReLU & Tanh',\
               'Leaky ReLU & Softmax', 'Tanh']
           
# Parameters
n_training = 20
mids = [0,1,2,3,4,5,6,7,8]
epochs = [500, 500, 250, 500, 500, 500, 250, 500, 500]
lrs = [0.00007, 0.0005, 0.0005, 0.0007, 0.0001, 0.0005, 0.0005, 0.00007, 0.0001]
weight_decays = [0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.001, 0.01, 0.001, 0.001]
learning_decays = [0.0001, 0, 0.01, 0, 0.0001, 0.001, 0.01, 0.001, 0.0001]
criterions = ['MSE', 'CROSSENTROPY', 'CROSSENTROPY', 'CROSSENTROPY', 'CROSSENTROPY',\
              'CROSSENTROPY', 'CROSSENTROPY', 'CROSSENTROPY', 'MSE']
learning_decay_types = ['exponential', 'exponential', 'time-based', 'exponential', \
                        'exponential', 'exponential', 'time-based', 'exponential', 'exponential']

# Load data
train_pts, test_pts, train_labels, test_labels, train_targets, test_targets = load_data()


# Uncomment this part if you would like to select the model to run 

"""
# Model selection

parser = argparse.ArgumentParser()

parser.add_argument("model", help="one of the following: RELU, RELU_SIGMOID, RELU_TANH, RELU_SOFTMAX, LEAKY_RELU, LEAKY_RELU_SIGMOID, LEAKY_RELU_TANH, LEAKY_RELU_SOFTMAX, TANH")

args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])


if args.model.upper() == 'RELU':
	history = train(model_selector(0, seed = 0), train_pts, train_targets, epochs[0], lrs[0], test_pts = test_pts, test_targets = test_targets, 
          train_history = True, test_history = True, weight_decay = weight_decays[0], lr_decay_type = learning_decay_types[0], lr_decay = learning_decays[0], 
          criterion = criterions[0], batch_size = 1000, optimizer = 'MiniBatch')
                        
elif args.model.upper() == 'RELU_SIGMOID':
	history = train(model_selector(1, seed = 0), train_pts, train_targets, epochs[1], lrs[1], test_pts = test_pts, test_targets = test_targets, 
          train_history = True, test_history = True, weight_decay = weight_decays[1], lr_decay_type = learning_decay_types[1], lr_decay = learning_decays[1], 
          criterion = criterions[1], batch_size = 1000, optimizer = 'MiniBatch')
          
elif args.model.upper() == 'RELU_TANH':
	history = train(model_selector(2, seed = 0), train_pts, train_targets, epochs[2], lrs[2], test_pts = test_pts, test_targets = test_targets, 
          train_history = True, test_history = True, weight_decay = weight_decays[2], lr_decay_type = learning_decay_types[2], lr_decay = learning_decays[2], 
          criterion = criterions[2], batch_size = 1000, optimizer = 'MiniBatch')
          
elif args.model.upper() == 'RELU_SOFTMAX':
	history = train(model_selector(3, seed = 0), train_pts, train_targets, epochs[3], lrs[3], test_pts = test_pts, test_targets = test_targets, 
          train_history = True, test_history = True, weight_decay = weight_decays[3], lr_decay_type = learning_decay_types[3], lr_decay = learning_decays[3], 
          criterion = criterions[3], batch_size = 1000, optimizer = 'MiniBatch')
          
elif args.model.upper() == 'LEAKY_RELU':
	history = train(model_selector(4, seed = 0), train_pts, train_targets, epochs[4], lrs[4], test_pts = test_pts, test_targets = test_targets, 
          train_history = True, test_history = True, weight_decay = weight_decays[4], lr_decay_type = learning_decay_types[4], lr_decay = learning_decays[4], 
          criterion = criterions[4], batch_size = 1000, optimizer = 'MiniBatch')
          
elif args.model.upper() == 'LEAKY_RELU_SIGMOID':
	history = train(model_selector(5, seed = 0), train_pts, train_targets, epochs[5], lrs[5], test_pts = test_pts, test_targets = test_targets, 
          train_history = True, test_history = True, weight_decay = weight_decays[5], lr_decay_type = learning_decay_types[5], lr_decay = learning_decays[5], 
          criterion = criterions[5], batch_size = 1000, optimizer = 'MiniBatch')
          
elif args.model.upper() == 'LEAKY_RELU_TANH':
	history = train(model_selector(6, seed = 0), train_pts, train_targets, epochs[6], lrs[6], test_pts = test_pts, test_targets = test_targets, 
          train_history = True, test_history = True, weight_decay = weight_decays[6], lr_decay_type = learning_decay_types[6], lr_decay = learning_decays[6], 
          criterion = criterions[6], batch_size = 1000, optimizer = 'MiniBatch')
          
elif args.model.upper() == 'LEAKY_RELU_SOFTMAX':
	history = train(model_selector(7, seed = 0), train_pts, train_targets, epochs[7], lrs[7], test_pts = test_pts, test_targets = test_targets, 
          train_history = True, test_history = True, weight_decay = weight_decays[7], lr_decay_type = learning_decay_types[7], lr_decay = learning_decays[7], 
          criterion = criterions[7], batch_size = 1000, optimizer = 'MiniBatch')
          
elif args.model.upper() == 'TANH':
	history = train(model_selector(8, seed = 0), train_pts, train_targets, epochs[8], lrs[8], test_pts = test_pts, test_targets = test_targets, 
          train_history = True, test_history = True, weight_decay = weight_decays[8], lr_decay_type = learning_decay_types[8], lr_decay = learning_decays[8], 
          criterion = criterions[8], batch_size = 1000, optimizer = 'MiniBatch')

print("\n",args.model.upper())
"""

# Running a single model 

print("Model with ReLU and Tanh activation functions:")
history = train(model_selector(2, seed = 0), train_pts, train_targets, epochs[2], lrs[2], test_pts = test_pts, test_targets = test_targets, 
          train_history = True, test_history = True, weight_decay = weight_decays[2], lr_decay_type = learning_decay_types[2], lr_decay = learning_decays[2], 
          criterion = criterions[2], batch_size = 1000, optimizer = 'MiniBatch')

from models import Module, Linear, Sequential, BatchNorm, ReLU, LeakyReLU, Tanh, Sigmoid
from torch import set_grad_enabled, clamp, matmul, Tensor, sum,\
ones, randn, empty, max, exp, addmm, log, zeros_like, zeros, manual_seed, stack, flatten, mean, var
from torch.nn.init import normal_
from random import uniform
from math import pi, sqrt
import math
import random

# Turn off autograd
set_grad_enabled(False)

# General constants
eps = 1e-5
momentum = 0.9

# Generates a data point sampled uniformly in [0,1]
# If the data is inside the disk centered at (0.5,0.5) of radius 1/2*pi, its label must be 1, else 0
def gen_point():
    center_x, center_y = 0.5, 0.5
    R = 1/sqrt(2*pi)
    rand_x, rand_y = uniform(0, 1), uniform(0, 1)
    label = 0
    if (rand_x - center_x) ** 2 + (rand_y - center_y) ** 2 <= R ** 2: #inside
        label = 1
    return [rand_x, rand_y], label

# Batch generator
def data_generator(pts, targets, labels, batch_size):
    data_len = pts.size(0)
    i = 0
    while i < data_len:
        j = i + batch_size if i +batch_size < data_len else data_len
        yield pts[i:j], targets[i:j], labels[i:j]
        i = j

# Generates N points
def gen_points(N):
    pts = []
    labels = []
    for _ in range(N):
        pt, label = gen_point()
        pts.append(pt)
        labels.append(label)
    return pts, labels

# One-hot encodes y
def convert_to_one_hot(y):
    y_onehot = empty(y.size(0), 2) #2 because boolean one hot
    y_onehot.zero_()
    y_onehot[range(y.size(0)), y.long()] = 1
    return Tensor(y_onehot)

# One-hot decodes y
def decode_one_hot(y_onehot):
    y = empty(y_onehot.size(0))
    y = Tensor([0. if x[0]==1 else 1. for x in y_onehot])
    return y

# Loads data
def load_data():
    # Generate 1k samples of training data + 1k samples of testing data
    all_points, all_labels = gen_points(2000)

    # Convert to tensors
    all_points = Tensor(all_points)               # points = datapoint (x and y position)
    all_labels = Tensor(all_labels)               # labels = 0 if the point is inside the disk, 1 if outside
    all_targets = convert_to_one_hot(all_labels)  # targets = one hot encodings of the labels

    # Split the dataset
    train_pts, test_pts = all_points[:1000], all_points[1000:]
    train_labels, test_labels = all_labels[:1000], all_labels[1000:]
    train_targets, test_targets = all_targets[:1000], all_targets[1000:]

    # Normalize the datasets
    mean,std = train_pts.mean(), train_pts.std()
    train_pts.sub_(mean).div_(std)
    test_pts.sub_(mean).div_(std)
    
    # Return data
    return train_pts, test_pts, train_labels, test_labels, train_targets, test_targets

# Computes the Mean Squared Error & Gradient at output
def mse(pred, target):
    return sum((pred - target)**2),  2* (pred-target)

# Softmax function stabilized by taking x = x-max(x)
def softmax(x):
    return exp(x-max(x))/sum(exp(x-max(x)))

# Computes the Cross Entropy Loss
def crossEntropyLoss(pred, target):
    return -sum(target*softmax(pred).log()), pred-target

# Instanciates a model with respect to the model id given as input to the function
def model_selector(mid, seed = 0):
    
    # ReLU activation
    if mid == 0:
        model = Sequential([Linear(2,25, method = 'random', seed = seed), ReLU(),
                            Linear(25,25, method = 'random', seed = seed + 1), ReLU(), 
                            Linear(25,25, method = 'random', seed = seed + 2), ReLU(), 
                            Linear(25,2, method = 'random', seed = seed + 3), ReLU()])
        return model
    
    # ReLU & Sigmoid activations
    elif mid == 1:
        model = Sequential([Linear(2,25, method = 'random', seed = seed), ReLU(),
                            Linear(25,25, method = 'random', seed = seed + 1), ReLU(), 
                            Linear(25,25, method = 'random', seed = seed + 2), ReLU(), 
                            Linear(25,2, method = 'random', seed = seed + 3), Sigmoid()])    
        return model
    
    # ReLU & Tanh activations
    elif mid == 2:
        model = Sequential([Linear(2,25, method = 'random', seed = seed), ReLU(),
                            Linear(25,25, method = 'random', seed = seed + 1), ReLU(), 
                            Linear(25,25, method = 'random', seed = seed + 2), ReLU(), 
                            Linear(25,2, method = 'random', seed = seed + 3), Tanh()])             
        return model     
    
    # ReLU & Softmax activations (included in the cross entropy)
    elif mid == 3:
        model = Sequential([Linear(2,25, method = 'random', seed = seed), ReLU(),
                            Linear(25,25, method = 'random', seed = seed + 1), ReLU(), 
                            Linear(25,25, method = 'random', seed = seed + 2), ReLU(), 
                            Linear(25,2, method = 'random', seed = seed + 3)])             
        return model  
    
    # Leaky ReLU activation
    elif mid == 4:
        model = Sequential([Linear(2,25, method = 'random', seed = seed), LeakyReLU(),#slope = 0.00005),
                            Linear(25,25, method = 'random', seed = seed + 1), LeakyReLU(),#slope = 0.00005), 
                            Linear(25,25, method = 'random', seed = seed + 2), LeakyReLU(),#slope = 0.00005), 
                            Linear(25,2, method = 'random', seed = seed + 3), LeakyReLU()])#slope = 0.00005)])             
        return model  
    
    # Leaky ReLU & Sigmoid activations
    elif mid == 5:
        model = Sequential([Linear(2,25, method = 'random', seed = seed), LeakyReLU(),
                            Linear(25,25, method = 'random', seed = seed + 1), LeakyReLU(), 
                            Linear(25,25, method = 'random', seed = seed + 2), LeakyReLU(), 
                            Linear(25,2, method = 'random', seed = seed + 3), Sigmoid()])             
        return model  
    
    # Leaky ReLU & Tanh activations
    elif mid == 6:
        model = Sequential([Linear(2,25, method = 'random', seed = seed), LeakyReLU(),
                            Linear(25,25, method = 'random', seed = seed + 1), LeakyReLU(), 
                            Linear(25,25, method = 'random', seed = seed + 2), LeakyReLU(), 
                            Linear(25,2, method = 'random', seed = seed + 3), Tanh()])             
        return model

    # Leaky ReLU & Softmax activations (included in the cross entropy)
    elif mid == 7:
        model = Sequential([Linear(2,25, method = 'random', seed = seed), LeakyReLU(),
                            Linear(25,25, method = 'random', seed = seed + 1), LeakyReLU(), 
                            Linear(25,25, method = 'random', seed = seed + 2), LeakyReLU(), 
                            Linear(25,2, method = 'random', seed = seed + 3)])             
        return model  
    
    # Tanh activations (included in the cross entropy)
    elif mid == 8:
        model = Sequential([Linear(2,25, method = 'random', seed = seed), Tanh(),
                            Linear(25,25, method = 'random', seed = seed + 1), Tanh(), 
                            Linear(25,25, method = 'random', seed = seed + 2), Tanh(), 
                            Linear(25,2, method = 'random', seed = seed + 3), Tanh()])             
        return model  
    
    # ReLU & Sigmoid activations with BatchNorm
    elif mid == 9:
        model = Sequential([Linear(2,25, method = 'random', seed = seed), BatchNorm(25), Tanh(),
                            Linear(25,25, method = 'random', seed = seed + 1), BatchNorm(25), Tanh(), 
                            Linear(25,25, method = 'random', seed = seed + 2), BatchNorm(25), Tanh(), 
                            Linear(25,2, method = 'random', seed = seed + 3), Tanh()])
        return model
    
# Trains the model over n epochs with mini batch processing
def train(model, train_pts, train_targets, n_epoch, lr0, test_pts = [], test_targets = [], 
          train_history = False, test_history = False, weight_decay = 0, lr_decay_type='exponential', lr_decay = 0, 
          criterion = 'MSE', batch_size = 1000, optimizer = 'MiniBatch'):
    
    # History
    model_history = dict()
    
    # Loss histories
    train_losses = []
    test_losses = []
    
    # Accuracy histories
    train_accs = []
    test_accs = []
    
    # Labels (needed if the accuracies are computed)
    train_labels = []
    test_labels = []
    if (train_history):
        train_labels = decode_one_hot(train_targets)
    if (test_history):
        test_labels = decode_one_hot(test_targets)
    
    # Train for n epochs
    for e in range(n_epoch):
        epoch_train_loss = 0 
        
        # Learning rate decay
        if lr_decay_type == 'exponential':
            lr = lr0 * math.exp( -lr_decay * n_epoch)
        if lr_decay_type == 'time-based':
            lr = lr0 * (1. / (1. + lr_decay * n_epoch))
        if lr_decay_type == 'step-decay':
            lr = lr0 * math.pow(0.5,math.floor((1+n_epoch)/10))

        # Generate batches
        generator = data_generator(train_pts, train_targets, train_labels, batch_size)
        
        # For each batch
        for i, (t_pts, ts, t_labels) in enumerate(generator):
            
            # Forward pass
            train_pred = model.forward(t_pts)

            # Compute loss
            if(criterion == 'MSE'):
                train_loss, grad_at_output = mse(train_pred, ts.view(-1, 2))
            elif(criterion == 'CROSSENTROPY'):
                train_loss, grad_at_output = crossEntropyLoss(train_pred, ts.view(-1, 2))
            else:
                raise NotImplementedError("Criterion not implemented")
                        
            # Weight decay
            weights = model.get_weights()
            l2_penality = 0
            for w in weights:
                l2_penality += (w**2).sum()
            train_loss += weight_decay * l2_penality/2
            
            # Backward prop
            model.backward(grad_at_output)             # grad_at_output += wd*w
            model.step(lr, weight_decay = weight_decay)
            epoch_train_loss += train_loss.item()
            
        # If we want to keep histories about training loss & accuracy               
        if (train_history):
                train_losses.append(epoch_train_loss)
                train_accs.append(compute_acc(model, t_pts, t_labels, train_pred)) 
                
        
        # If we want to keep histories about testing loss & accuracy
        if (test_history):
            
            # Forward pass
            test_pred = model.forward(test_pts)
            
            # Compute loss
            if(criterion=='MSE'):
                test_loss, _ = mse(test_pred, test_targets.view(-1, 2))
            elif(criterion=='CROSSENTROPY'):
                test_loss, _ = crossEntropyLoss(test_pred, test_targets.view(-1, 2))
            else:
                raise NotImplementedError("Criterion not implemented")
                
            test_losses.append(test_loss.item())
            test_accs.append(compute_acc(model, test_pts, test_labels, test_pred))
            
        print("epoch = ", e, "; train_loss = ", epoch_train_loss, "; test_loss = ", test_loss.item(), "; train_accuracy = ", compute_acc(model, t_pts, t_labels, train_pred), "; test_accuracy = ", compute_acc(model, test_pts, test_labels, test_pred))
        
    # If we want to keep histories about training
    if (train_history):
        model_history['train_loss'] = train_losses
        model_history['test_loss'] = test_losses
    
    # If we want to keep histories about testing
    if (test_history):
        model_history['train_accuracy'] = train_accs
        model_history['test_accuracy'] = test_accs
        
    return model_history

# Computes accuracy
def compute_acc(model, pts, labels, pred):
    n_samples = pts.size(0)
    _, indices = max(pred.view(-1,2), 1)
    accuracy = (sum(indices.view(-1,1) == labels.view(-1, 1)) / float(n_samples) * 100).item()
    return accuracy
    

from torch import set_grad_enabled, clamp, matmul, Tensor, sum,\
ones, randn, empty, max, exp, addmm, log, zeros_like, zeros, manual_seed, stack, flatten, mean, var
from torch.nn.init import normal_
from random import uniform
from math import pi, sqrt
import math
import random

# Interface
class Module (object):
    def __init__(self):
        self.x = None           # The input to the current module
        self.prev_grad = None   # Gradient at the output of this module
    
    # Forward-pass method
    def forward (self, x):
        raise NotImplementedError
        
    # Backward propagation method
    def backward (self, prev_grad):
        raise NotImplementedError
        
    # Update method
    def step(self, lr, weight_decay):
        raise NotImplementedError
        
    def get_weights(self):
        raise NotImplementedError
        
        
# Fully connected layer
class Linear(Module):
    def __init__(self, in_dim, out_dim, method = 'random', seed = 0):
        super().__init__()
        
        self.vw = zeros((in_dim, out_dim))
        self.vb = zeros((1, out_dim))
        
        # Fix the seed
        manual_seed(seed)
        
        # Scale with 0.2 not to have inf values and losses  
        if method == 'random':
            self.w = randn(in_dim, out_dim) * 0.2 # Random weights from normal dist
            self.b = randn(1, out_dim) * 0.2      # Random bias from normal dist

        elif method == 'he':
            self.w = randn(in_dim, out_dim)*sqrt(2/in_dim) * 0.2
            self.b = randn(1, out_dim) * 0.2

        elif method == 'xavier':
            self.w = randn(in_dim, out_dim)*sqrt(1/in_dim) * 0.1
            self.b = randn(1, out_dim) * 0.1

        elif method == 'zeros':
            self.w = zeros((in_dim, out_dim))
            self.b = zeros((1, out_dim))

        else:
            raise NotImplementedError("Initialization method not implemented")
        
    def forward(self, x):
        self.x = x
        return addmm(self.b, x, self.w)        # Without an activation function
    
    def backward(self, prev_grad): # add wd as parameter in all the backward
        self.prev_grad = prev_grad             # Cache the grad at the output for weight update.      
        current_grad = (prev_grad @ self.w.t()) #.add_(wd, self.w)   
        return current_grad                                          
        
    def step(self, lr, weight_decay, optimizer='SGD'):
        w = lr * (self.x.t() @ self.prev_grad) - lr * (weight_decay * self.w) # Multiply the grad at output with dy/dw to update the weights
        b = lr * (self.prev_grad.sum(dim=0))      # dy/db = 1. then dL / db = dy, which is the grad at the output
        if optimizer == 'SGD':
            self.w -= w
            self.b -= b
        elif optimizer == 'Adagrad':
            self.vw += w**2
            self.vb += b**2
            self.w -= w/torch.sqrt(self.vw+eps)
            self.b -= b/torch.sqrt(self.vb+eps)
        elif optimizer == 'RMSProp':
            self.vw = momentum*self.vw + (1-momentum)*(w**2)
            self.vb = momentum*self.vb + (1-momentum)*(b**2)
            self.w -= w/torch.sqrt(self.vw+eps)
            self.b -= b/torch.sqrt(self.vb+eps)
        elif optimizer == 'momentum':
            self.vw = momentum*self.vw + lr*w
            self.vb = momentum*self.vb + lr*b
            self.w -= self.vw
            self.b -= self.vb
        else:
            raise NotImplementedError("Optimizer not implemented")
    def get_weights(self):
        return self.w
    
    
# Sequential structure to combine several modules
class Sequential(Module):
    def __init__(self, module_list):
        self.module_list = module_list
        
    def forward(self, x):
        out = self.module_list[0].forward(x)
        for module in self.module_list[1:]:
            out = module.forward(out)
        return out
    
    def backward(self, gradwrtoutput):
        for model in self.module_list[::-1]:
            gradwrtoutput = model.backward(gradwrtoutput)
            
    def step(self, lr, weight_decay = 0):
        for module in self.module_list:
            module.step(lr, weight_decay)
            
    def get_weights(self):
        weights = []
        for module in self.module_list:
            m_weight = module.get_weights()
            if (m_weight == None):
                continue
            weights += [module.get_weights()]
        return weights
    
    
# Batch Normalization layer
class BatchNorm(Module):
    
    def __init__(self, dim):
        self.dim = dim
        self.gamma = ones(dim) # scale param
        self.beta = zeros(dim) # shift param
        self.mean = zeros(dim) # averaged mean
        self.var = zeros(dim) # averaged variance
        super().__init__()
        
    def forward(self, x, mode):
        self.x = x
        if mode == 'train':
            self.mean = momentum * self.mean + (1 - momentum) * mean(x)
            self.var = momentum * self.var + (1 - momentum) * var(x) 
            self.centered = x - mean(x)
            self.std = sqrt(var(self.x) + eps)
            self.norm = self.centered / self.std # normalized value
            return self.gamma * self.norm + self.beta # normalized, scaled and shifted value
        elif mode == 'test':
            return self.gamma * (x - self.mean) / sqrt(self.var + eps) + self.beta
            
    def backward(self, prev_grad):
        self.prev_grad = prev_grad
        dgamma = sum(prev_grad * self.norm)
        dbeta = sum(prev_grad)
        dnorm = prev_grad * self.gamma # scaled value
        curr_grad = 1/self.dim / self.std * (self.dim * dnorm - sum(dnorm) - self.norm * sum(dnorm * self.norm))    
        #return curr_grad, dgamma, dbeta # gamma and beta to update in step
        return curr_grad
        
    def step(self, lr, weight_decay, optimizer):
        pass
    
    def get_weights(self):
        pass
    
    
# ReLu activation function
class ReLU(Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        self.x = x
        return x.clamp(min = 0)       # Negative values -> 0, positive values stay the same (relu = max(0,x))
    
    def backward(self, prev_grad):
        self.prev_grad = prev_grad    # Cache the gradient at the output 
        curr_grad = prev_grad.clone() # Copy it for returning the new gradient
        curr_grad[self.x < 0] = 0     # Relu = max(0,x), grad_relu = 0 if x <= 0 else 1
        return curr_grad
    
    def step(self, lr, weight_decay):
        pass    
    
    def get_weights(self):
        pass
    
    
    
# Leaky ReLu activation function
class LeakyReLU(Module):
    
    def __init__(self, slope = 0.01):
        self.slope = slope
        super().__init__()
    
    def forward(self, x):
        self.x = x
        return max(self.slope * x, x)
    
    def backward(self, prev_grad):
        self.prev_grad = prev_grad    # Cache the gradient at the output 
        curr_grad = prev_grad.clone()
        curr_grad[self.x < 0] *= self.slope
        return curr_grad
    
    def step(self, lr, weight_decay):
        pass    
    
    def get_weights(self):
        pass
    
    
# Tanh activation function
class Tanh(Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        self.x = x
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))                         # Tanh function
    
    def backward(self, prev_grad):
        self.prev_grad = prev_grad                                             # Cache the gradient at the output
        tanh_x = (exp(self.x) - exp(-self.x)) / (exp(self.x) + exp(-self.x))   # Tanh of x
        curr_grad = prev_grad * (1 - (tanh_x)**2)                              # Current gradient 
        return curr_grad
    
    def step(self, lr, weight_decay):
        pass

    def get_weights(self):
        pass
    
    
# Sigmoid activation function
class Sigmoid(Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        self.x = x
        return 1 / (1 + exp(-x))                              # Sigmoid function
    
    def backward(self, prev_grad):
        self.prev_grad = prev_grad                            # Cache the gradient at the output 
        sigmoid_x = 1 / (1 + exp(-self.x))                    # Sigmoid of x
        curr_grad = prev_grad * sigmoid_x * (1 - sigmoid_x)   # Current gradient
        return curr_grad

    def step(self, lr, weight_decay):
        pass
    
    def get_weights(self):
        pass
    
    

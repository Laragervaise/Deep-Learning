import os
from torchvision import datasets
import torch
from models import *
import matplotlib.pyplot as plt

def generate_pair_sets(nb):
    data_dir = os.environ.get('PYTORCH_DATA_DIR')
    if data_dir is None:
        data_dir = './data'

    train_set = datasets.MNIST(data_dir + '/mnist/', train = True)
    train_input = train_set.data.view(-1, 1, 28, 28).float()
    train_target = train_set.targets

    test_set = datasets.MNIST(data_dir + '/mnist/', train = False)
    test_input = test_set.data.view(-1, 1, 28, 28).float()
    test_target = test_set.targets

    return mnist_to_pairs(nb, train_input, train_target) + \
           mnist_to_pairs(nb, test_input, test_target)


def mnist_to_pairs(nb, input, target):
    input = torch.functional.F.avg_pool2d(input, kernel_size = 2)
    a = torch.randperm(input.size(0))
    a = a[:2 * nb].view(nb, 2)
    input = torch.cat((input[a[:, 0]], input[a[:, 1]]), 1)
    classes = target[a]
    target = (classes[:, 0] <= classes[:, 1]).long()
    return input, target, classes

def load():

    # Load the data
    size = 1000
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(size)

    #normalization
    #check https://stats.stackexchange.com/questions/174823/
    mu, std = train_input.mean(), train_input.std()
    train_input, test_input = train_input.sub(mu).div(std), test_input.sub(mu).div(std)

    #split the images
    train_input1, train_input2 = train_input[:, 0, :, :], train_input[:, 1, :, :] 
    test_input1, test_input2 = test_input[:, 0, :, :], test_input[:, 1, :, :] 

    #split the number pairs
    train_classes1, train_classes2 = train_classes[:, 0], train_classes[:, 1]
    test_classes1, test_classes2 = test_classes[:, 0], test_classes[:, 1]
    
    inputs = [train_input1, train_input2, test_input1, test_input2, train_classes1, train_classes2, test_classes1, test_classes2, train_target, test_target]
    
    return inputs



def data_generator(input1, input2, digits1, digits2, targets, batch_size):
    data_len = input1.size(0)
    i = 0
    while i < data_len:
        j = i + batch_size if i +batch_size < data_len else data_len
        yield input1[i:j], input2[i:j], \
              digits1[i:j],digits2[i:j],\
              targets[i:j]
        i = j  


"""
:param model_constructor: constructor for the model
:param optimizer_name: 'sgd' or 'adam'
:param lr: learning rate
:param batch_size: batch_size.
:return: an encapsulated model ready for the training
"""
def model_selector(model_constructor, optimizer_name, lr, batch_size, weight_decay=0, nb_epochs=25):
    
    model = dict()
    model['model'] = model_constructor()
    model['criterion'] = nn.CrossEntropyLoss()
    model['nb_epochs'] = nb_epochs
    model['batch_size'] = batch_size
    if(optimizer_name == 'sgd'):
        model['optimizer'] = torch.optim.SGD(model['model'].parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        model['optimizer'] =  torch.optim.Adam(model['model'].parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
    return model


def initialize_models():
    model_FNN = model_selector(FNN, 'adam', 0.014, 64, weight_decay = 0.001, nb_epochs = 50)
    model_FNN_WS = model_selector(FNN_WS, 'adam', 0.013 , 64, weight_decay = 0.01, nb_epochs=50)
    model_FNN_WS_AUX = model_selector(FNN_WS_AUX, 'adam', 0.014, 64, weight_decay = 0,nb_epochs=50 )
    model_FNN_AUX = model_selector(FNN_AUX, 'adam', 0.012 ,64, weight_decay = 0, nb_epochs = 50)

    model_CNN = model_selector(CNN, 'adam', 0.01, 64, weight_decay = 0.001, nb_epochs = 50)
    model_CNN_WS_AUX = model_selector(CNN_WS_AUX, 'adam', 0.014 , 64, weight_decay = 0.001, nb_epochs = 50)
    model_CNN_WS = model_selector(CNN_WS, 'adam', 0.014 , 64, weight_decay = 0.001, nb_epochs = 50)
    model_CNN_AUX = model_selector(CNN_AUX, 'adam', 0.013, 64, weight_decay = 0, nb_epochs = 50)
    
    models = [model_FNN, model_FNN_WS, model_FNN_AUX, model_FNN_WS_AUX, model_CNN, model_CNN_WS, model_CNN_AUX, model_CNN_WS_AUX]
    
    return models

"""
:param model: a dict encapsulating the model and its properties
:param input1_tr: lhs of image pairs from training set
:param input2_tr: rhs of image pairs from training set
:param input1_te: lhs of image pairs from test set
:param input2_te: rhs of image pairs from test set
:param digits1_tr: classes of input1 from training set
:param digits2_tr: classes of input2 from training set
:param digits1_te: classes of input1 from test set
:param digits2_te: classes of input2 from test set
:param targets_tr: final boolean value indicating whether lhs <= rhs from training set
:param targets_te: final boolean value indicating whether lhs <= rhs from test set
:return: a dict encapsulating the model history
"""
def train(model, input1_tr, input2_tr, digits1_tr, digits2_tr, targets_tr, \
                 input1_te =None, input2_te =None, digits1_te=None, digits2_te=None, targets_te=None):
    
    epochs = model['nb_epochs']
    batch_size = model['batch_size']
    criterion = model['criterion']
    optimizer = model['optimizer']
    mdl = model['model']
      
    #  a dict to return whatever value we want to return
    #  e.g. loss at each epoch (useful for plotting)
    model_history = dict()
    
    
    train_loss_history = [] #a list to keep track of the losses at each epoch
    test_loss_history = []
    test_acc_history = []
    
    for e in range(epochs):
        
        epoch_train_loss = 0
        generator = data_generator(input1_tr, input2_tr, digits1_tr, digits2_tr, targets_tr, batch_size)
        for input1, input2, digits1, digits2, targets in generator:
            d1, d2, pred = mdl(input1, input2)   # run through the network
            pred_loss =  criterion(pred.view(-1, 2), targets) # loss due to boolean value
     
            if d1 is not None:
                pred_loss += criterion(d1.view(-1, 10), digits1)
                pred_loss += criterion(d2.view(-1, 10), digits2)
                pred_loss /= 3
                
            loss = pred_loss.item() #magnitude of the loss
            epoch_train_loss += loss
            mdl.zero_grad()         #reset the gradients for this epoch
            pred_loss.backward()    #calculate the gradients
            optimizer.step()        #update the weights
             
        train_loss_history.append(epoch_train_loss) #record the train loss 
        print('epoch=', e, 'train_loss=', epoch_train_loss, end = ' ')
        
        if input1_te is not None:
            epoch_test_loss = 0
            test_generator = data_generator(input1_te, input2_te, digits1_te, digits2_te, targets_te, batch_size)
            with torch.no_grad():
                for input1, input2, digits1, digits2, targets in test_generator:
                    d1, d2, pred = mdl(input1, input2)
                    pred_loss =  criterion(pred.view(-1, 2), targets)

                    if d1 is not None:
                        pred_loss += criterion(d1.view(-1, 10), digits1)
                        pred_loss += criterion(d2.view(-1, 10), digits2)
                        pred_loss /= 3

                    loss = pred_loss.item()
                    epoch_test_loss += loss
        

            test_loss_history.append(epoch_test_loss) #record the test loss
            
            acc_target, _, _= compute_nb_errors(mdl, input1_te, input2_te, digits1_te, digits2_te, targets_te)
            test_acc_history.append(acc_target)
            print('test_loss=', epoch_test_loss,'test_accuracy=', acc_target, end = '')

        print()        
        
        
    model_history['train_loss_history'] = train_loss_history
    model_history['test_loss_history'] = test_loss_history
    model_history['test_acc_history'] = test_acc_history
    return model_history

"""
:param model: a dict encapsulating the model and its properties
:param input1: lhs of image pairs
:param input2: rhs of image pairs
:param digits1: classes of input1 
:param digits2: classes of input2
:param targets: final boolean value indicating whether lhs <= rhs
:return: a triplet indicating the accuracies ordered as (boolean,lhs,rhs)
"""
def compute_nb_errors(model, input1, input2, digits1, digits2, targets):
    n_samples = input1.shape[0]
    with torch.no_grad():
        d1,d2,pred = model(input1, input2)           # predict the digits + boolean
        _, indices = torch.max(pred.view(-1,2), 1) # torch.max returns the max value from the distribution and its corresponding index
        acc_target = (sum(indices == targets) / float(n_samples) * 100).item()  #calculate accuracy

        acc_d1, acc_d2 = 0, 0
        if d1 is not None: #the model returns digits if it makes use of aux loss. in this case we can report the accuracy of predicting the digits.
            _, indices1 = torch.max(d1.view(-1,10), 1)
            _, indices2 = torch.max(d2.view(-1,10), 1)
            acc_d1 += (sum(indices1 == digits1) / float(n_samples) * 100).item()
            acc_d2 += (sum(indices2 == digits2) / float(n_samples) * 100).item()


        return (acc_target, acc_d1, acc_d2)

  

"""
:param input1: lhs of image pairs
:param input2: rhs of image pairs
:param digits1: classes of input1 
:param digits2: classes of input2
:param targets: final boolean value indicating whether lhs <= rhs
:return: a triplet indicating the accuracies ordered as (boolean,lhs,rhs)
"""
def cross_val_score(input1, input2, digits1, digits2, targets, model_constructor, optimizer_name, lr, batch_size, weight_decay,nb_epoch, k_folds=3):
    len_train = input1.shape[0]
    indices = list(range(len_train))
    random.seed(8)
    random.shuffle(indices)
    acc_target, acc_d1, acc_d2  = 0,0,0
    for k in range(k_folds):
        model = model_selector(model_constructor, optimizer_name, lr, batch_size, weight_decay, nb_epoch)  # init the same model
        val_indices = indices[k*len_train//k_folds:(k+1)*len_train//k_folds] # 1 validation fold
        train_indices = list(set(indices) - set(val_indices))                # k-1 training fold
        
        #train the model with k-1 training fold
        history = train(model, input1[train_indices], input2[train_indices], digits1[train_indices], digits2[train_indices], targets[train_indices])
        
        #compute the accuracy on 1 validation fold
        accs = compute_nb_errors(model['model'], input1[val_indices], input2[val_indices], digits1[val_indices], digits2[val_indices], targets[val_indices])
        
        acc_target += accs[0]
        acc_d1 += accs[1]
        acc_d2 += accs[2]
        #print('fold=', k, ' loss = ', history['loss_history'][-1])
    return (acc_target / k_folds, acc_d1 /k_folds, acc_d2 /k_folds)



def grid_search():
    models = [FNN, FNN_WS, FNN_WS_AUX, FNN_AUX, CNN, CNN_WS_AUX, CNN_WS, CNN_AUX]
    epochs = [50]
    batch_sizes = [64]
    weight_decays = [0, 0.1, 0.01, 0.001]
    lrs = [0.001 * x for x in range(1, 15)]
    opts = ['adam']

    for m in models:
        best_batch, best_lr, best_opt, best_wd, best_epoch, best_acc = None, None, None, None,None,None
        for b in batch_sizes:
            for lr in lrs:
                for opt in opts:
                    for wd in weight_decays:
                        for e in epochs:
                            start = timeit.default_timer()
                            acc_t, acc_d1, acc_d2 = cross_val_score(train_input1, train_input2, train_classes1, train_classes2,  train_target, m, opt,lr,b,wd, e, k_folds=3)
                            end = timeit.default_timer()
                            print(m, b, e, wd, round(lr, 3), opt, '/// accuracies: ', acc_t, acc_d1, acc_d2, '/// time: ', round(end - start, 2))
                            if best_acc is None or acc_t > best_acc:
                                best_epoch, best_wd, best_batch, best_lr, best_opt, best_acc = e, wd, b, lr, opt, acc_t
        print('for model ', m, 'best accuracy is ', best_acc, 'at ', 'lr=', best_lr, 'b=', best_batch, 'opt=', best_opt, 'wd=', best_wd, 'epoch=', best_epoch)

def plotLoss(mean_histories, label):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    train_mean_loss = [hist['train_loss_history'] for hist in mean_histories]
    test_mean_loss = [hist['test_loss_history'] for hist  in mean_histories]
    
    
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    
    for i, (mean_train, mean_test) in enumerate(zip(train_mean_loss, test_mean_loss)):
        
        ax.plot(mean_train, alpha=0.5, color = colors[i], linewidth = 2.0, label = label)

        ax.plot(mean_test, alpha=0.5, color = colors[i], linewidth = 2.0, label = label + ' Test', linestyle = 'dashed')

    ax.legend(loc='best', facecolor = 'white')
    ax.set_ylabel('Loss')
    ax.set_xlabel('# of Epochs')
    plt.show()


def plotacc(mean_histories, line_label):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    test_mean_acc = [hist['test_acc_history'] for hist  in mean_histories]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    
    for i, mean_test_acc in enumerate(test_mean_acc):

        ax.plot(mean_test_acc, alpha=0.5, color = colors[i], linewidth = 2.0, label = line_label)
        #ax.fill_between(range(len(mean_test_acc)), mean_test_acc - std_test_acc, mean_test_acc + std_test_acc, color = colors[i], alpha=0.2)

    ax.legend(loc='best', facecolor = 'white')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('# of Epochs')
    ax.set_ylim(0,100)
    plt.show()

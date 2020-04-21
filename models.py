# Fully connected neural network
import torch
import torch.nn as nn

# WS+AUX
class FNN_WS_AUX(torch.nn.Module):
    def __init__(self):
        super(FNN_WS_AUX, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(196,196), 
            nn.ReLU(),
            nn.Linear(196,10), 
            nn.Softmax(dim=1)
        )
        
    def forward(self, im1, im2):
        d1 = self.model(im1.view(-1, 196))
        d2 = self.model(im2.view(-1, 196))

        one_hot = torch.nn.functional.one_hot( (( torch.argmax(d1,1) ) <= ( torch.argmax(d2,1) )).long(), num_classes=2).float()
        #create one_hot [0,1] or [1,0] depending on d1 <=d2
        #.long is necessary because otherwise it was becoming bool
        #.float is necessary because loss function works with float.
        return d1, d2, one_hot
    
# WS+ no aux
class FNN_WS(torch.nn.Module):
    def __init__(self):
        super(FNN_WS, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(196,196),
            nn.ReLU(),
            nn.Linear(196,10), 
            nn.Softmax(dim=1)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(20, 2),
            nn.Softmax(dim = 1)
        )
        
        
    def forward(self, im1, im2):
        d1 = self.model(im1.view(-1, 196))
        d2 = self.model(im2.view(-1, 196))

        d1d2 = torch.cat([d1,d2], dim = 1)
        target = self.predictor(d1d2)
        return None, None, target
    
# no WS+ aux
class FNN_AUX(torch.nn.Module):
    def __init__(self):
        super(FNN_AUX, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(196,196), #1*14*14 -> 10*8*8
            nn.ReLU(),
            nn.Linear(196,10), #1*14*14 -> 10*8*8
            nn.Softmax(dim=1)
        )
        
        self.model2 = nn.Sequential(
            nn.Linear(196,196), #1*14*14 -> 10*8*8
            nn.ReLU(),
            nn.Linear(196,10), #1*14*14 -> 10*8*8
            nn.Softmax(dim=1)
        )
        
        
    def forward(self, im1, im2):
        d1 = self.model(im1.view(-1, 196))
        d2 = self.model2(im2.view(-1, 196))

        one_hot = torch.nn.functional.one_hot( (( torch.argmax(d1,1) ) <= ( torch.argmax(d2,1) )).long(), num_classes=2).float()

        return d1, d2, one_hot
    
# no WS+ no aux
class FNN(torch.nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(196,196), #1*14*14 -> 10*8*8
            nn.ReLU(),
            nn.Linear(196,10), #1*14*14 -> 10*8*8
            nn.Softmax(dim=1)
        )
        
        self.model2 = nn.Sequential(
            nn.Linear(196,196), #1*14*14 -> 10*8*8
            nn.ReLU(),
            nn.Linear(196,10), #1*14*14 -> 10*8*8
            nn.Softmax(dim=1)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(20, 2),
            nn.Softmax(dim = 1)
        )
        
        
        
    def forward(self, im1, im2):
        d1 = self.model(im1.view(-1, 196))
        d2 = self.model2(im2.view(-1, 196))

        d1d2 = torch.cat([d1,d2], dim = 1)
        target = self.predictor(d1d2)
        return None, None, target
    
    
# WS+AUX
class CNN_WS_AUX(torch.nn.Module):
    def __init__(self):
        super(CNN_WS_AUX, self).__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size =3), #1*14*14 -> 32*12 12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride=2), #   12 -> 6
            nn.Conv2d(16, 32, kernel_size = 3, ), #4
            nn.MaxPool2d(kernel_size = 2, stride=2), # 32*2*2   
        )
        
        self.linear_layer = nn.Sequential(
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )
        
        
    def forward(self, im1, im2):
        im1 = self.conv_layer(im1.view(-1, 1, 14, 14))
        d1 = self.linear_layer(im1.view(-1, 128))

        im2 = self.conv_layer(im2.view(-1, 1, 14, 14))
        d2 = self.linear_layer(im2.view(-1, 128))
        
        one_hot = torch.nn.functional.one_hot( (( torch.argmax(d1,1) ) <= ( torch.argmax(d2,1) )).long(), num_classes=2).float()

        return d1, d2, one_hot
    
# WS+ NO AUX
class CNN_WS(torch.nn.Module):
    def __init__(self):
        super(CNN_WS, self).__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size =3), #1*14*14 -> 32*12 12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride=2), #   12 -> 6
            nn.Conv2d(16, 32, kernel_size = 3, ), #4
            nn.MaxPool2d(kernel_size = 2, stride=2), # 32*2*2   
        )
        
        self.linear_layer = nn.Sequential(
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(20, 2),
            nn.Softmax(dim = 1)
        )
        
    def forward(self, im1, im2):
        im1 = self.conv_layer(im1.view(-1, 1, 14, 14))
        d1 = self.linear_layer(im1.view(-1, 128))

        im2 = self.conv_layer(im2.view(-1, 1, 14, 14))
        d2 = self.linear_layer(im2.view(-1, 128))
        
        
        
        d1d2 = torch.cat([d1,d2], dim = 1)
        target = self.predictor(d1d2)
        return None, None, target
    
# no WS+AUX
class CNN_AUX(torch.nn.Module):
    def __init__(self):
        super(CNN_AUX, self).__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size =3), #1*14*14 -> 32*12 12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride=2), #   12 -> 6
            nn.Conv2d(16, 32, kernel_size = 3, ), #4
            nn.MaxPool2d(kernel_size = 2, stride=2), # 32*2*2   
        )
        
        self.linear_layer = nn.Sequential(
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )
        
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size =3), #1*14*14 -> 32*12 12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride=2), #   12 -> 6
            nn.Conv2d(16, 32, kernel_size = 3, ), #4
            nn.MaxPool2d(kernel_size = 2, stride=2), # 32*2*2   
        )
        
        self.linear_layer2 = nn.Sequential(
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )
        
    def forward(self, im1, im2):
        im1 = self.conv_layer(im1.view(-1, 1, 14, 14))
        d1 = self.linear_layer(im1.view(-1, 128))

        im2 = self.conv_layer2(im2.view(-1, 1, 14, 14))
        d2 = self.linear_layer2(im2.view(-1, 128))
        
        one_hot = torch.nn.functional.one_hot( (( torch.argmax(d1,1) ) <= ( torch.argmax(d2,1) )).long(), num_classes=2).float()

        return d1, d2, one_hot
    
    
# no WS+ no AUX
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size =3), #1*14*14 -> 32*12 12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride=2), #   12 -> 6
            nn.Conv2d(16, 32, kernel_size = 3, ), #4
            nn.MaxPool2d(kernel_size = 2, stride=2), # 32*2*2   
        )
        
        self.linear_layer = nn.Sequential(
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )
        
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size =3), #1*14*14 -> 32*12 12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride=2), #   12 -> 6
            nn.Conv2d(16, 32, kernel_size = 3, ), #4
            nn.MaxPool2d(kernel_size = 2, stride=2), # 32*2*2   
        )
        
        self.linear_layer2 = nn.Sequential(
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(20, 2),
            nn.Softmax(dim = 1)
        )

        
    def forward(self, im1, im2):
        im1 = self.conv_layer(im1.view(-1, 1, 14, 14))
        d1 = self.linear_layer(im1.view(-1, 128))

        im2 = self.conv_layer2(im2.view(-1, 1, 14, 14))
        d2 = self.linear_layer2(im2.view(-1, 128))
        
        d1d2 = torch.cat([d1,d2], dim = 1)
        target = self.predictor(d1d2)
        return None, None, target
## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # Conv layers
        # input size 1x227x227
        self.conv1 = nn.Conv2d(1, 96, kernel_size=4, stride=4, padding=0)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        
        # Max Pooling layer 
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Linear layers (last layer output is 136 values, 2 for each of the 68 keypoint (x, y) pairs)
        self.fc1 = nn.Linear(6*6*256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2*68)
        
        # Dropout Layers
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.4)
        self.dropout3 = nn.Dropout(p=0.6)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(256, eps=1e-05)
        self.bn2 = nn.BatchNorm2d(384, eps=1e-05)
        self.bn3 = nn.BatchNorm2d(384, eps=1e-05)
        self.bn4 = nn.BatchNorm2d(256, eps=1e-05)
        self.bn5 = nn.BatchNorm1d(96, eps=1e-05)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # Glorot (also known as Xavier) Initialization
                m.weight = nn.init.xavier_uniform(m.weight, gain=1)
           
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # Conv layers followed by regularization techniques to avoid overfitting (max pooling, dropout layers and batch normalization)
        
        x = F.elu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = F.elu(self.conv2(x))
        x = self.bn1(x)
        x = self.pool(x)
        
        x = F.elu(self.conv3(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        
        x = F.elu(self.conv4(x))
        x = self.bn3(x)
        x = self.dropout2(x)
        
        x = F.elu(self.conv5(x))
        x = self.bn4(x)
        x = self.pool(x)


        # Flatten layer to collapse the spatial dimensions of the input into the channel dimension
        x = x.view(x.size(0), -1) 
        
        # Fully connected layers with ELU (Exponential Linear Unit) activation function
        x = F.elu(self.fc1(x))
        x = self.bn5(x)
        x = self.dropout3(x)
        
        x = F.elu(self.fc2(x))
        x = self.bn5(x)
        x = self.dropout3(x)
        
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

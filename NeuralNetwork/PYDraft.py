#!/usr/bin/env python
# coding: utf-8

# In[16]:


import io
import torch
import torch.nn as nn
import math
from collections import OrderedDict
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np


# In[17]:


# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[18]:


################## AlexNet ##################
def bn_relu(inplanes):
    return nn.Sequential(nn.BatchNorm2d(inplanes), nn.ReLU(inplace=True))

def bn_relu_pool(inplanes, kernel_size=3, stride=2):
    return nn.Sequential(nn.BatchNorm2d(inplanes), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=kernel_size, stride=stride))

class AlexNet(nn.Module):
    def __init__(self, num_classes=5):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, bias=False)
        self.relu_pool1 = bn_relu_pool(inplanes=96)
        self.conv2 = nn.Conv2d(96, 192, kernel_size=5, padding=2, groups=2, bias=False)
        self.relu_pool2 = bn_relu_pool(inplanes=192)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1, groups=2, bias=False)
        self.relu3 = bn_relu(inplanes=384)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2, bias=False)
        self.relu4 = bn_relu(inplanes=384)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2, bias=False)
        self.relu_pool5 = bn_relu_pool(inplanes=256)
        # classifier
        self.conv6 = nn.Conv2d(256, 256, kernel_size=5, groups=2, bias=False)
        self.relu6 = bn_relu(inplanes=256)
        self.conv7 = nn.Conv2d(256, num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_pool1(x)
        x = self.conv2(x)
        x = self.relu_pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu_pool5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = x.view(x.size(0), -1)
        return x


# In[19]:


net = AlexNet()
net.load_state_dict(torch.load('./pytorch-models/cnn.pth'))
net.eval()


# In[20]:


# Scaling of data
valdir = './data/1/test16morph.txt'
mylines = []                             # Declare an empty list named mylines.
with open(valdir, 'rt') as myfile: # Open lorem.txt for reading text data.
    for myline in myfile:                # For each line, stored as myline,
        mylines.append(float(myline.split()[1]))       # add its contents to mylines.
print(mylines)   

min_val = min(mylines)
max_val = max(mylines)
lower_scale = 0
upper_scale = 4

for i in range(len(mylines)):
    mylines[i] = int(round(((upper_scale-lower_scale)*((mylines[i]-min_val)/(max_val-min_val)))+lower_scale))
    
print(mylines)


# In[21]:


def read_img(root, filedir, transform=None):
    # Data loading
    with open(filedir, 'r') as f:
        lines = f.readlines()  
    output = []    
    for line in lines:
        linesplit = line.split('\n')[0].split(' ')
        #print(linesplit)
        addr = linesplit[0]
        target = torch.Tensor([float(linesplit[1])])
        img = Image.open(os.path.join(root, addr)).convert('RGB')

        if transform is not None:
            img = transform(img)
        
        output.append([img, target])

    return output


# In[25]:


def main():
    # net definition 
    #net = AlexNet()
    # net = Nets.ResNet(block = Nets.BasicBlock, layers = [2, 2, 2, 2], num_classes = 1).cuda()

    # load pretrained model
    #load_model(torch.load('AlexNet.pth', map_location=torch.device('cpu')), net) #load_model('pytorch-models/alexnet.pth')
    # load_model(torch.load('./models/resnet18.pth'), net)

    # evaluate
    #net.eval()

    # loading data...
    #root = 'C:/Users/Lenovo/Documents/DTU-AP/SCUT-FBP5500_v2.1/SCUT-FBP5500_v2/Images'
    #valdir = './data/1/test1.txt'
    #root = 'C:/Users/Lenovo/Documents/DTU-AP/Multi-Morph/asian/af/asian_female_16'
    #valdir = './data/1/morph16.txt'
    #valdir = './data/1/test1.txt'
    root = 'C:/Users/Lenovo/Documents/AdvancedProject/NeuralNetwork/data/morph16'
    valdir = './data/1/test16morph.txt'
    
    print('hér')
    print(root, valdir)
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])  
    val_dataset = read_img(root, valdir, transform=transform)
    print(val_dataset)
    
    with torch.no_grad():
        label = []
        pred = []

        for i, (img, target) in enumerate(val_dataset):
            img = img.unsqueeze(0)#.cuda(non_blocking=True)
            target = target#.cuda(non_blocking=True)
            output = net(img).squeeze(1)
            label.append(target.cpu()[0])
            pred.append(output.cpu()[0])
            print(i)

        # measurements
        label = np.array(label)
        pred = np.array(pred)
        correlation = np.corrcoef(label, pred)[0][1]
        mae = np.mean(np.abs(label - pred))
        rmse = np.sqrt(np.mean(np.square(label - pred)))
    
    print('\n Label Array: ' + str(label))
   # print('Label Mean: ' + str(label.mean()) + '\n')
    print('Prediction Array: ' + str(pred))
    print('Prediction Mean: ' + str(pred.mean()) + '\n')
    
    #print('Correlation:{correlation:.4f}\t'
    #      'Mae:{mae:.4f}\t'
    #      'Rmse:{rmse:.4f}\t'.format(
    #        correlation=correlation, mae=mae, rmse=rmse))


if __name__ == '__main__':
    main()
    


# In[ ]:




